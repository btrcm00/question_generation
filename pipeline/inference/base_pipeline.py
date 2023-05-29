import math
import regex as re
import torch
import time
import logging
import onnx
import onnxruntime as ort

from transformers import AutoTokenizer
from multiprocessing import Process, Manager
from queue import Queue
from threading import Thread, Event

from common.common_keys import *
from common.config import *
from common.utils import ModelUtils, pre_process, timer
from pipeline.trainer.model.bartpho import BartPhoPointer


class BaseSampler:
    def __init__(self, config: PipelineConfig, **kwargs):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parallel_input_processing = config.sampling_parallel_input_processing

        self.best_checkpoint_path = self.get_best_checkpoint(self.config.training_output_dir,
                                                             checkpoint_type=self.config.pipeline_checkpoint_type)
        self.model, self.tokenizer = self.load_checkpoint(model_class=BartPhoPointer)
        self.model_utils = ModelUtils(input_max_length=self.config.pipeline_input_max_length, tokenizer=self.tokenizer)

    def get_best_checkpoint(self, folder_checkpoint: str = None, checkpoint_type: str = "best"):
        """get checkpoint with highest BLEU score in `folder_checkpoint`

        Args:
            folder_checkpoint (str, optional): _description_. Defaults to None.

        Returns:
            str: path of best checkpoint
        """
        if checkpoint_type not in ["best", "last"]:
            self.logger.error(f'>>>`checkpoint_type` must be in ["best", "last"], not {checkpoint_type}<<<')
            # raise ValueError(f'`checkpoint_type` must be in ["best", "last"], not {checkpoint_type}')

        _re_checkpoint = re.compile(r"^checkpoint\-(\d+)$")
        if _re_checkpoint.search(folder_checkpoint.split("/")[-1]):
            return folder_checkpoint
        best_checkpoint_until_now = ""
        checkpoints = [path for path in os.listdir(folder_checkpoint) if _re_checkpoint.search(path) is not None and \
                       os.path.isdir(os.path.join(folder_checkpoint, path))]
        while True:
            if len(checkpoints) == 0:
                return folder_checkpoint
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            if checkpoint_type == "last":
                return os.path.join(folder_checkpoint, last_checkpoint)
            state_of_last_checkpoint = os.path.join(folder_checkpoint, last_checkpoint) + "/trainer_state.json"
            if not os.path.isfile(state_of_last_checkpoint):
                checkpoints.remove(last_checkpoint)
                continue
            best_checkpoint_until_now = json.load(open(state_of_last_checkpoint, "r", encoding="utf8"))[
                "best_model_checkpoint"]
            break
        return best_checkpoint_until_now if best_checkpoint_until_now is not None else folder_checkpoint

    @timer
    def load_checkpoint(self, model_class: str):
        if not self.best_checkpoint_path:
            self.logger.error(">>>NOT FIND MODEL WITH INPUT PATH<<<")
            self.best_checkpoint_path = self.config.pipeline_pretrained_path

        self.logger.info(f"LOADING CHECKPOINT AT {self.best_checkpoint_path} ...")
        if self.config.pipeline_onnx:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            if self.config.pipeline_device != "cpu":
                providers.append("CUDAExecutionProvider")
            qg_model = ort.InferenceSession(self.best_checkpoint_path, opts, providers=providers)
        else:
            qg_model = model_class.from_pretrained(self.best_checkpoint_path, model_config=self.config)
        qg_model.to(self.config.pipeline_device)
        qg_model.eval()
        qg_tokenizer = AutoTokenizer.from_pretrained(self.best_checkpoint_path)
        return qg_model, qg_tokenizer

    def __collate_fn(self, data):
        batch = {}
        for feature_key in data[0].keys():
            # prepare for tag feature ids
            if feature_key == ENTITY_WEIGHT:
                batch[feature_key] = torch.vstack([ele[feature_key].to(self.config.pipeline_device) for ele in data])
            # input ids vs attention mask
            else:
                batch[feature_key] = torch.stack(
                    [ele[feature_key][:self.config.pipeline_input_max_length].to(self.config.pipeline_device) for ele in
                     data])
        return batch

    def pre_process_input(self, passage_ans_clue: str, question_type: str = None):
        types = question_type.upper().replace(" ", "_") if question_type else None
        _passage = passage_ans_clue
        _passage = " ".join([" ".join(ele).replace("< ", "<").replace(" >", ">").replace("/ ", "/") \
                             for ele in Config.vncore_nlp.tokenize(_passage.replace("_", " "))])
        passage = f"<{types}> {_passage}" if types is not None else _passage
        processed_passage = self.model_utils.truncate_passage(passage=passage)
        return processed_passage

    def _inference(self, processed_passages: list, num_beams: int = 1, num_return_sequences: int = 1):
        """_summary_

        Args:
            processed_passages (list): passage input to model
            num_beams (int, optional): _description_. Defaults to 1.
            num_return_sequences (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if num_return_sequences > num_beams:
            self.logger.error(
                f">>> `num_return_sequences` has to be smaller or equal to `num_beams`., but {num_return_sequences} < {num_beams} <<<")
            # raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
        output = []
        num_iters = math.ceil(len(processed_passages) / self.config.sampling_inference_batch_size)
        self.logger.info(f"INFERENCE {num_iters} batch")
        for i in range(num_iters):
            tokenized_passages = self.tokenizer(
                processed_passages[
                i * self.config.sampling_inference_batch_size:(i + 1) * self.config.sampling_inference_batch_size],
                padding="max_length",
                max_length=self.config.pipeline_input_max_length,
                return_tensors='pt',
                truncation=True)

            decoded_preds = self._generate(tokenized_passages, num_beams=num_beams,
                                           num_return_sequences=num_return_sequences)
            output += [decoded_preds[i:i + num_return_sequences] for i in
                       range(0, len(decoded_preds), num_return_sequences)]

        return output

    def _generate(self, tokenized_passages, num_beams: int = 1, num_return_sequences: int = 1):
        inputs = self.__collate_fn([
            {
                INPUT_IDS: tokenized_passages.input_ids[idx],
                ATTENTION_MASK: tokenized_passages.attention_mask[idx],
                ENTITY_WEIGHT: torch.tensor([0])
            }
            for idx in range(len(tokenized_passages.input_ids))
        ])
        model_output = self.model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences,
                                           no_repeat_ngram_size=2,
                                           max_length=self.config.pipeline_output_max_length)
        decoded_preds = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return decoded_preds

    # @timer
    def get_input_sampling_with_entity(self, tokenized_passage, output_lst: list, is_segmented: bool = True):
        ner_passage = [" ".join(e) for e in tokenized_passage]
        ner_dict, processed_passage = self.model_utils.get_entity_from_passage(ner_passage,
                                                                               is_segmented_list=is_segmented)
        output_lst.insert(0, ner_dict)
        for ent, ner_lst in ner_dict.items():
            if not any(w and w[0].isupper() for w in ent.split("_")) and not re.search(r"\d", ent) and ner_lst[0] in [
                "CARDINAL"]:
                continue

            ques_type_mapping = Config.ques_type_config.get("NER", {})
            ques_type_list = ques_type_mapping.get(ner_lst[0], [])

            for ques_type in ques_type_list:
                types = ques_type.upper().replace(" ", "_")
                passage_ans_clue = self.model_utils.prepare_model_input(passage=processed_passage, answer=ent,
                                                                        ans_lst=ner_lst, ques_type=types)
                _id = self.model_utils.generate_hash(f"{ent}-{processed_passage}")
                example = BaseQGData(
                    id=_id,
                    passage=passage_ans_clue,
                    answer=ent,
                    question_type=types
                )
                output_lst.append(example)

    # @timer
    def get_input_sampling_with_pos(self, tokenized_passage, output_lst: list = None, is_segmented: bool = True):
        ques_type_mapping = Config.ques_type_config.get("POS", {})

        all_tag_lst = []
        for lst in ques_type_mapping.values():
            all_tag_lst += lst

        chunk_of_answer_lst = self.model_utils.get_chunk(tokenized_passage, tag_lst=all_tag_lst,
                                                         is_segmented=is_segmented)
        pos_passage = " ".join(" ".join(ele) for ele in tokenized_passage).split()
        _pos_passage = " ".join(pos_passage)
        for _type, tag_lst in ques_type_mapping.items():
            ques_type = _type.upper().replace(" ", "_")
            chunk_of_answer = [chunk for chunk in chunk_of_answer_lst if
                               any(chunk[1].upper().startswith(t) for t in tag_lst)]
            for ans in chunk_of_answer:
                answer = " ".join(ans[2])
                answer_start_idx = len(" ".join(pos_passage[:ans[-2]]))
                # if start index == 0, not add 1
                # else add 1 (1 space)
                if ans[-2] != 0:
                    answer_start_idx += 1
                ans_lst = ["ANS", answer_start_idx, answer_start_idx + len(answer)]
                passage_ans_clue = self.model_utils.prepare_model_input(passage=_pos_passage, answer=answer,
                                                                        ans_lst=ans_lst, ans_type="ANS",
                                                                        ques_type=ques_type)
                _id = self.model_utils.generate_hash(f"{answer}-{_pos_passage}")
                example = BaseQGData(
                    id=_id,
                    passage=passage_ans_clue,
                    answer=answer,
                    question_type=ques_type
                )
                output_lst.append(example)

    def _run_input_sampling_parallel(self, passage: str):
        tokenized_passage = self.model_utils.tokenize_passage(passage=passage, depth=0)
        tokenized_passage = [e if e[-1] == "." else e + ["."] for e in tokenized_passage if len(e) > 1]
        is_segmented = True
        proc = []
        input_func = [self.get_input_sampling_with_entity, self.get_input_sampling_with_pos]
        manager = Manager()
        output = manager.list()
        for fn in input_func:
            p = Process(target=fn, args=(tokenized_passage, output, is_segmented))
            p.start()
            proc.append(p)
        for p in proc:
            p.join()
        return output, tokenized_passage

    def _sampling(self, input_lst: List[BaseQGData], entity_dict: Dict = None, original: Dict = None,
                  num_beams: int = 1,
                  num_return_sequences: int = 1):
        self.logger.info(
            f"START INFERENCE {len(input_lst)} examples with batch_size {self.config.sampling_inference_batch_size} ...")
        start_gen = time.time()
        passage_lst = [e.passage for e in input_lst]
        predict_pointer = self._inference(processed_passages=passage_lst, num_return_sequences=num_return_sequences,
                                          num_beams=num_beams)
        self.logger.info(f"Gen TIME: {time.time() - start_gen}")
        samplings = []
        for idx, qg_example in enumerate(input_lst):
            qg_example.question = predict_pointer[idx]
            _example = BaseSamplingData(data=qg_example)

            if self.config.sampling_verify:
                _example.verified = self.model_utils.question_validation(data=qg_example)
            samplings.append(_example)
        return SamplingData(sampling=samplings,
                            entity_dict=entity_dict if self.config.sampling_return_entity else None,
                            original=original[idx] if original else "")


class QuestionSampler(BaseSampler):
    def __init__(self, config: PipelineConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.processed_queue = Queue(maxsize=10000)
        self.output_queue = Queue(maxsize=10000)
        self.input_queue = Queue(maxsize=10000)
        self.sampling_event = Event()
        self.sampling_event.clear()

        self.input_process_threads = [Thread(target=self.run_input_sampling_parallel, args=(), daemon=True) for _ in
                                      range(5)]
        self.inference_thread = Thread(target=self.inference, args=(), daemon=True)
        self.output_thread = None

    def start(self):
        [thread.start() for thread in self.input_process_threads]
        self.inference_thread.start()
        self.output_thread.start()

    def join(self):
        [thread.join() for thread in self.input_process_threads]
        self.output_thread.join()
        self.inference_thread.join()

    def create_output_thread(self, target_func=None, args=None):
        if target_func is not None:
            if args is None:
                args = ()
            self.output_thread = Thread(target=target_func, args=args, daemon=None)

    # @timer
    def inference(self, num_beams: int = 1, num_return_sequences: int = 1):
        """_summary_

        Args:
            processed_passages (list): passage input to model
            num_beams (int, optional): _description_. Defaults to 1.
            num_return_sequences (int, optional): _description_. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if num_return_sequences > num_beams:
            self.logger.error(
                f">>> `num_return_sequences` has to be smaller or equal to `num_beams`., but {num_return_sequences} < {num_beams} <<<")
            # raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
        while not self.sampling_event.is_set():
            if self.processed_queue.empty():
                continue
            samples, _id = self.processed_queue.get()
            output = self._inference([sample.data.passage for sample in samples.sampling], num_beams=num_beams,
                                     num_return_sequences=num_return_sequences)
            for idx, sample in enumerate(samples.sampling):
                sample.data.question = output[idx]
            self.output_queue.put((samples, _id))

    @timer
    def predict(self, sample: dict, num_beams: int = 1, num_return_sequences: int = 1):
        """_summary_

        Args:
            sample (dict): _description_
            num_beams (int, optional): _description_. Defaults to 1.
            num_return_sequences (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        input_passage = self.pre_process_input(passage_ans_clue=sample[MODEL_INPUT],
                                               question_type=sample[MODEL_QUESTION_TYPE_INPUT])
        return BaseQGData(
            id=sample.get(ID, self.model_utils.generate_hash(f"{input_passage}")),
            passage=input_passage,
            answer="",
            question=self._inference([input_passage], num_beams=num_beams, num_return_sequences=num_return_sequences)[
                0],
            question_type=sample[MODEL_QUESTION_TYPE_INPUT]
        )

    def run_input_sampling_parallel(self):
        while not self.sampling_event.is_set():
            if self.input_queue.empty():
                continue
            passage, _id = self.input_queue.get()
            output, _ = self._run_input_sampling_parallel(passage=passage)
            ner_dict = output.pop(0)
            output = [BaseSamplingData(data=example) for example in output]
            output = SamplingData(sampling=output, entity_dict=ner_dict)
            self.processed_queue.put((output, _id))

    @pre_process
    @timer
    def sampling(self, passage: str, num_beams: int = 1, num_return_sequences: int = 1):
        """Sampling examples from passage

        Args:
            passage (str): passage to sampling
            num_return_sequences (int, optional): number of returned output. Defaults to 1.
            num_beams (int, optional): using in beam search. Defaults to 1.
            _id (str):
        """
        self.logger.info("START SAMPLING ...")
        # input_lst, answer_lst = []
        if self.parallel_input_processing:
            output, tokenized_passage = self._run_input_sampling_parallel(passage)
            entity_dict = output.pop(0)
            # for ele in output:
            #     if ele.answer in answer_lst or any(ele.answer in _pre for _pre in answer_lst):
            #         continue
            #     input_lst.append(ele)
            #     answer_lst.append(ele.answer)

        processed_passage = " ".join(" ".join(ele) for ele in tokenized_passage)
        return self._sampling(input_lst=output, entity_dict=entity_dict, num_return_sequences=num_return_sequences,
                              num_beams=num_beams), processed_passage


if __name__ == "__main__":
    pass
