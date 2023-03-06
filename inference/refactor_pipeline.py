import math
import os
import json
import regex as re
import torch

from transformers import AutoTokenizer
from tqdm import tqdm
from queue import Queue
from multiprocessing import Process, Manager
from threading import Thread, Event

from common.common_keys import *
from common.config import Config, ModelInputTag
from common.utils import ModelUtils, pre_process, timer
from model.bartpho import BartPhoPointer


class QuestionSampler:
    def __init__(
            self,
            input_max_length: int = None,
            output_max_length: int = None,
            folder_checkpoint: str = None,
            model_type=None,
            inference_batch_size: int = 4,
            model_device: str = "cpu",
            return_entity: bool = False,
            need_verify: bool = False,
            parallel_input_processing: bool = False,
            num_of_threads: int = 4
    ):
        """_summary_

        Args:
            input_max_length (int, optional): max length of model input. Defaults to None.
            folder_checkpoint (str, optional): folder contain checkpoint to load pretrained. Defaults to None.
        """

        self.device = model_device
        self.return_entity = return_entity
        self.need_verify = need_verify
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.parallel_input_processing = parallel_input_processing
        self.inference_batch_size = inference_batch_size
        self.best_checkpoint_path = self.get_best_checkpoint(folder_checkpoint)
        if not self.best_checkpoint_path:
            raise "NOT FIND MODEL WITH INPUT PATH"
        self.model, self.tokenizer = self.load_checkpoint(model_type=model_type,
                                                          pretrained_path=self.best_checkpoint_path)
        self.model_utils = ModelUtils(input_max_length=input_max_length, tokenizer=self.tokenizer)
        self.num_of_threads = num_of_threads

    @staticmethod
    def get_best_checkpoint(folder_checkpoint: str = None):
        """get checkpoint with highest BLEU score in `folder_checkpoint`

        Args:
            folder_checkpoint (str, optional): _description_. Defaults to None.

        Returns:
            str: path of best checkpoint
        """
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
            state_of_last_checkpoint = os.path.join(folder_checkpoint, last_checkpoint) + "/trainer_state.json"
            if not os.path.isfile(state_of_last_checkpoint):
                checkpoints.remove(last_checkpoint)
                continue
            best_checkpoint_until_now = json.load(open(state_of_last_checkpoint, "r", encoding="utf8"))[
                "best_model_checkpoint"]
            break
        return best_checkpoint_until_now if best_checkpoint_until_now is not None else folder_checkpoint

    @timer
    def load_checkpoint(self, model_type, pretrained_path: str):
        print(f"LOADING CHECKPOINT AT {pretrained_path} ...")
        if model_type == BartPhoPointer:
            qg_model = model_type.from_pretrained(pretrained_path, model_config={"use_pointer": True})
        else:
            qg_model = model_type.from_pretrained(pretrained_path, training_config={"use_pointer": True})

        qg_model.to(self.device)
        qg_model.eval()
        qg_tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        return qg_model, qg_tokenizer

    def __collate_fn(self, data):
        batch = {}
        for feature_key in data[0].keys():
            # prepare for tag feature ids
            if feature_key == ENTITY_WEIGHT:
                batch[feature_key] = torch.vstack([ele[feature_key].to(self.device) for ele in data])
            # input ids vs attention mask
            else:
                batch[feature_key] = torch.stack(
                    [ele[feature_key][:self.input_max_length].to(self.device) for ele in data])
        return batch

    @timer
    def inference(self, processed_passages: list, num_beams: int = 1, num_return_sequences: int = 1):
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
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
        output = []
        num_iters = math.ceil(len(processed_passages) / self.inference_batch_size)
        for i in tqdm(range(num_iters)):
            passage_tokenized = self.tokenizer(
                processed_passages[i * self.inference_batch_size:(i + 1) * self.inference_batch_size],
                padding="max_length",
                max_length=self.input_max_length,
                return_tensors='pt')
            inputs = self.__collate_fn([
                {
                    INPUT_IDS: passage_tokenized.input_ids[idx],
                    ATTENTION_MASK: passage_tokenized.attention_mask[idx],
                    ENTITY_WEIGHT: torch.tensor([0])
                }
                for idx in range(len(passage_tokenized.input_ids))
            ])

            model_output = self.model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences,
                                               no_repeat_ngram_size=2, max_length=self.output_max_length)
            decoded_preds = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

            output += [[decoded_preds[i + j] for j in range(num_return_sequences)] for i in
                       range(0, len(decoded_preds), num_return_sequences)]

        return output

    def pre_process_input(self, passage_ans_clue: str, question_type: str):
        types = question_type.upper().replace(" ", "_")
        _passage = passage_ans_clue
        if ModelInputTag.clue not in _passage:
            annotated_passage = [" ".join(ele).replace("< ", "<").replace(" >", ">").replace("/ ", "/") \
                                 for ele in Config.vncore_nlp.tokenize(_passage.replace("_", " "))]

            _passage = " ".join(
                f"{ModelInputTag.clue} {ele} {ModelInputTag.close_clue}".replace(f". {ModelInputTag.close_clue}",
                                                                                 f"{ModelInputTag.close_clue} .") \
                    if self.model_utils.special_pattern.search(ele) else ele for ele in annotated_passage)
        else:
            _passage = " ".join([" ".join(ele).replace("< ", "<").replace(" >", ">").replace("/ ", "/") \
                                 for ele in Config.vncore_nlp.tokenize(_passage.replace("_", " "))])
        processed_passage = self.model_utils.truncate_passage(passage=f"<{types}> {_passage}")
        return processed_passage

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
        return {
            "passage": input_passage,
            "predict question": self.inference([input_passage], num_beams=num_beams,
                                               num_return_sequences=num_return_sequences)[0],
        }

    @timer
    def get_input_sampling_with_entity(self, tokenized_passage, output_lst: list, is_segmented: bool = True):
        passage_lst, answer_lst, ques_type_lst = [], [], []

        ner_passage = [" ".join(e) for e in tokenized_passage]
        ner_dict, processed_passage = self.model_utils.get_entity_from_passage(ner_passage,
                                                                               is_segmented_list=is_segmented)
        output_lst.insert(0, ner_dict)
        for ent, ner_lst in ner_dict.items():
            if not any(w and w[0].isupper() for w in ent.split("_")) and not re.search(r"\d", ent):
                continue

            ques_type_mapping = Config.ques_type_config.get("NER", {})
            ques_type_list = ques_type_mapping.get(ner_lst[0], [])

            for ques_type in ques_type_list:
                types = ques_type.upper().replace(" ", "_")
                # passage_ans_clue = self.model_utils.prepare_model_input(passage=processed_passage, answer=ent,
                #                                                         ans_lst=ner_lst, ques_type=types)
                if self.parallel_input_processing:
                    # output_lst.append({
                    #     PASSAGE: passage_ans_clue,
                    #     ANSWER: ent,
                    #     QUESTION_TYPE: types
                    # })
                    output_lst.append({
                        PASSAGE: processed_passage,
                        ANSWER: ent,
                        "ans_lst": ner_lst,
                        QUESTION_TYPE: types
                    })
                else:
                    passage_lst.append(processed_passage)
                    answer_lst.append(ent)
                    ques_type_lst.append(types)
        if not self.parallel_input_processing:
            return passage_lst, answer_lst, ques_type_lst, ner_dict

    @timer
    def get_input_sampling_with_pos(self, tokenized_passage, output_lst: list = None, is_segmented: bool = True):
        passage_lst, answer_lst, ques_type_lst = [], [], []

        ques_type_mapping = Config.ques_type_config.get("POS", {})
        all_tag_lst = []
        for lst in ques_type_mapping.values():
            all_tag_lst += lst
            
        chunk_of_answer_lst, pos_passage = self.model_utils.get_chunk(tokenized_passage, tag_lst=all_tag_lst,
                                                                    is_segmented=is_segmented)
        for _type, tag_lst in ques_type_mapping.items():
            ques_type = _type.upper().replace(" ", "_")
            chunk_of_answer = [chunk for chunk in chunk_of_answer_lst if chunk[1].upper() in tag_lst]
            
            for ans in chunk_of_answer:
                answer = " ".join(ans[2])
                answer_start_idx = len(" ".join(pos_passage[:ans[-2]]))
                # if start index == 0, not add 1
                # else add 1 (1 space)
                if ans[-2] != 0:
                    answer_start_idx += 1
                ans_lst = ["ANS", answer_start_idx, answer_start_idx + len(answer)]
                # passage_ans_clue = self.model_utils.prepare_model_input(passage=" ".join(pos_passage), answer=answer,
                #                                                         ans_lst=ans_lst, ans_type="ANS",
                #                                                         ques_type=ques_type)
                if self.parallel_input_processing:
                    # output_lst.append({
                    #     PASSAGE: " ".join(pos_passage),
                    #     ANSWER: answer,
                    #     QUESTION_TYPE: ques_type
                    # })
                    output_lst.append({
                        PASSAGE: " ".join(pos_passage),
                        ANSWER: answer,
                        "ans_lst": ans_lst,
                        QUESTION_TYPE: ques_type
                    })
                else:
                    passage_lst.append(" ".join(pos_passage))
                    answer_lst.append(answer)
                    ques_type_lst.append(ques_type)
        if not self.parallel_input_processing:
            return passage_lst, answer_lst, ques_type_lst

    def run_input_sampling_parallel(self, tokenized_passage, is_segmented: bool = True):
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

        return output

    def testttt(self, input_dict):
        output = []
        input_queue = Queue(maxsize=1000)
        bar = tqdm(total=len(input_dict), initial=0, leave=True)

        event = Event()
        event.clear()

        threads = [Thread(target=self.model_utils.prepare_model_input_threading, args=(bar, output, input_queue, event), daemon=True)
                   for _ in range(40)]

        [thread.start() for thread in threads]

        for ele in input_dict:
            input_queue.put(ele)

        event.set()
        [thread.join() for thread in threads]
        return output
    
    def input_sampling_with_thread(self, tokenized_passage, is_segmented: bool = True):
        output = []

        input_dict = self.run_input_sampling_parallel(tokenized_passage=tokenized_passage, is_segmented=is_segmented)
        entity_dict = input_dict.pop(0)
        print(f"{len(input_dict)} examples")
        output = self.testttt(input_dict)
        print(4444444444444)
        # input_queue = Queue(maxsize=1000)
        # bar = tqdm(total=len(input_dict), initial=0, leave=True)

        # event = Event()
        # event.clear()

        # threads = [Thread(target=self.model_utils.prepare_model_input_threading, args=(bar, output, input_queue, event), daemon=True)
        #            for _ in range(self.num_of_threads)]

        # [thread.start() for thread in threads]

        # for ele in input_dict:
        #     input_queue.put(ele)

        # event.set()
        # [thread.join() for thread in threads]

        return output, entity_dict

    @pre_process
    @timer
    def sampling(self, passage: str, num_beams: int = 1, num_return_sequences: int = 1, _id: str = None):
        """Sampling examples from passage

        Args:
            passage (str): passage to sampling
            num_return_sequences (int, optional): number of returned output. Defaults to 1.
            num_beams (int, optional): using in beam search. Defaults to 1.
            _id (str):
        """
        print("START SAMPLING ...")
        tokenized_passage = self.model_utils.tokenize_passage(passage=passage, depth=0)
        if len(tokenized_passage) > 5000:
            return []
        tokenized_passage = [e if e[-1] == "." else e + ["."] for e in tokenized_passage if len(e) > 1]
        is_segmented = True

        passage_lst, answer_lst, ques_type_lst = [], [], []
        if self.parallel_input_processing:
            output, entity_dict = self.input_sampling_with_thread(tokenized_passage, is_segmented=is_segmented)
            # entity_dict = output.pop(0)
            for ele in output:
                passage_lst.append(ele[PASSAGE])
                answer_lst.append(ele[ANSWER])
                ques_type_lst.append(ele[QUESTION_TYPE])
        else:
            ner_passage, ner_answer, ner_ques_type, entity_dict = self.get_input_sampling_with_entity(tokenized_passage,[],
                                                                                                      is_segmented=is_segmented)
            pos_passage, pos_answer, pos_ques_type = self.get_input_sampling_with_pos(tokenized_passage, [],
                                                                                      is_segmented=is_segmented)
            passage_lst = ner_passage + pos_passage
            answer_lst = ner_answer + pos_answer
            ques_type_lst = ner_ques_type + pos_ques_type
            entity_dict = entity_dict

        return self.base_sampling(_id=_id, passage_lst=passage_lst, answer_lst=answer_lst,
                                  ques_type_lst=ques_type_lst, entity_dict=entity_dict,
                                  num_return_sequences=num_return_sequences, num_beams=num_beams)

    def base_sampling(self, _id: str, passage_lst: list, ques_type_lst: list, answer_lst: list,
                      entity_dict: dict = None, original: list = None, num_beams: int = 1,
                      num_return_sequences: int = 1):
        assert len(passage_lst) == len(answer_lst) == len(ques_type_lst)
        if original:
            assert len(original) == len(passage_lst)
        if len(passage_lst) == 0:
            return []

        samplings = []
        print(f"START INFERENCE {len(passage_lst)} examples ...")
        # start_gen = time.time()
        predict_pointer = self.inference(processed_passages=passage_lst, num_return_sequences=num_return_sequences,
                                         num_beams=num_beams)
        # print('Gen TIME: ', time.time() - start_gen)

        for idx, p in enumerate(passage_lst):
            example = {
                EXAMPLE_ID: _id,
                PASSAGE: p,
                ANSWER: answer_lst[idx],
                PREDICT_QUESTION: predict_pointer[idx],
                QUESTION_TYPE: ques_type_lst[idx],
                ORIGINAL: original[idx] if original else ""
            }
            if self.return_entity:
                example[MODEL_ENTITY_DICT_INPUT] = entity_dict
            verify = None
            if self.need_verify:
                verify = self.model_utils.question_validation(passage=p, question=predict_pointer[idx],
                                                              answer=answer_lst[idx])
            example[VERIFIED] = verify
            samplings.append(example)
        return samplings


if __name__ == "__main__":
    pass
