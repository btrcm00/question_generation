import argparse
import os

from queue import Queue
from tqdm import tqdm
from transformers import AutoTokenizer
from threading import Thread, Event

from common.utils import *
from common.config import *
from common.constants import *


class DatasetConstructor(metaclass=SingletonMeta):
    def __init__(self, constructor_config: PipelineConfig = None):
        self.config = constructor_config if constructor_config is not None else PipelineConfig()
        self.logger = logging.getLogger(__name__)
        tokenizer = AutoTokenizer.from_pretrained(self.config.pipeline_pretrained_path)

        self.model_utils = ModelUtils(input_max_length=self.config.pipeline_input_max_length,
                                      tokenizer=tokenizer)
        self.output_folder = f"{self.config.pipeline_dataset_folder}/processed"
        check_exist_folder(self.output_folder)
        assert sum(self.config.constructor_ratio) == 1, "Sum of ratio must be 1"

        input_data_folder = f"{self.config.pipeline_dataset_folder}/raw"
        assert os.path.isdir(input_data_folder)
        self.qa_dataset = self.load_qa_dataset(data_path=input_data_folder)
        self.processed_data = self.load_processed_data()
        self.temp_data_file = self.get_temp_file()
        self.init()

    def init(self):
        self.input_queue = Queue(maxsize=10000)
        self.process_bar = tqdm(total=len(self.qa_dataset))
        self.process_event = Event()
        self.process_event.clear()

    def get_temp_file(self):
        prefix_temp_filename = "all_examples_"
        all_files = os.listdir(self.output_folder)
        all_files = [f.replace(".pkl", "") for f in all_files if "all" in f]
        all_files.sort(key=lambda f: int(f.replace(prefix_temp_filename, "")))
        indice = int(all_files[-1].replace(prefix_temp_filename, "")) + 1 if all_files else 0

        return f"{prefix_temp_filename}{indice}.pkl"

    def load_processed_data(self):
        file_lst = [os.path.join(self.output_folder, f) for f in os.listdir(self.output_folder) if
                    os.path.isfile(os.path.join(self.output_folder, f)) and ".pkl" in f]
        processed_data = []
        for f in file_lst:
            processed_data += [ele[ID] for ele in load_file(f)]
        processed_data = list(set(processed_data))
        self.logger.info(f"LOADED {len(processed_data)} processed examples")
        return processed_data

    def load_qa_dataset(self, data_path, mode: str = None):
        if mode is None:
            mode = ""

        file_lst = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                    os.path.isfile(os.path.join(data_path, f)) and mode in f]
        dataset = []
        for f in file_lst:
            dataset += json.load(open(f, "r", encoding="utf8"))  # ["data"]
        self.logger.info(f"Loaded {len(dataset)} examples")

        return dataset

    def process_data(self, output: list):
        while not self.process_event.is_set() or not self.input_queue.empty():
            data = self.input_queue.get()
            if data.id not in self.processed_data and data.answer["text"] and data.answer["text"][0] in data.context:
                answer = " ".join(
                    [" ".join([e for e in ele]) for ele in Config.vncore_nlp.tokenize(data.answer["text"][0])])
                answer = tone_normalization(answer)
                while answer.endswith(tuple(string.punctuation + string.whitespace)):
                    answer = answer[:-1]

                ner_dict, processed_passage = self.model_utils.get_entity_from_passage(passage=data.context,
                                                                                       is_segmented_list=False)
                ner_ans = "ANS"
                ner_lst = []
                for ent in ner_dict.keys():
                    if answer in ent:
                        ner_ans = ner_dict[ent][0]
                        ner_lst = ner_dict[ent]
                if not ner_lst:
                    idx_of_answer = processed_passage.find(answer)
                    if idx_of_answer == -1:
                        ner_lst = []
                    else:
                        ner_lst = [ner_ans, idx_of_answer, idx_of_answer + len(answer)]
                if ner_lst and processed_passage:
                    if ner_ans not in self.model_utils.special_tokens:
                        self.model_utils.special_tokens += [ner_ans, "/" + ner_ans]
                    question_type = get_question_style(passage=data.question)
                    passage_ans_clue = self.model_utils.prepare_model_input(passage=processed_passage, answer=answer,
                                                                            ans_lst=ner_lst, ques_type=question_type)
                    ques_ans_label = self.model_utils.bert_prepare_data(
                        data.question)  # + f" <{ner_ans}> {answer} </{ner_ans}>")

                    if all(e in passage_ans_clue for e in
                           [ner_ans, "/" + ner_ans]) and "{hl" not in passage_ans_clue:
                        if passage_ans_clue:
                            output.append({
                                ID: data.id,
                                ANSWER: answer,
                                MODEL_INPUT: passage_ans_clue,
                                MODEL_LABEL: ques_ans_label,
                                MODEL_QUESTION_TYPE_INPUT: question_type,
                                MODEL_ENTITY_DICT_INPUT: ner_dict
                            })
                    if len(output) % 500 == 0 and len(output) > 0:
                        save_file(output, f"{self.output_folder}/{self.temp_data_file}")

            self.process_bar.update(1)

    def save_dataset(self, dataset: list, shuffle: bool = True):
        _lst = []
        clean_data = dataset
        # for data in tqdm(dataset):
        #     if data[ID] not in _lst:
        #         clean_data.append(data)
        #         _lst.append(data[ID])
        if shuffle:
            random.shuffle(clean_data)

        l = len(clean_data)
        ratio = [sum(self.config.constructor_ratio[:i]) for i in range(1, len(self.config.constructor_ratio))]
        train_data = clean_data[:int(l * ratio[0])]
        dev_data = clean_data[int(l * ratio[0]):int(l * ratio[1])]
        test_data = clean_data[int(l * ratio[1]):]

        save_file(train_data, self.output_folder + "/train_examples.pkl")
        save_file(dev_data, self.output_folder + "/dev_examples.pkl")
        save_file(test_data, self.output_folder + "/test_examples.pkl")

    def run(self):
        output = []
        threads = [Thread(target=self.process_data, args=(output), daemon=True) for _ in
                   range(self.config.constructor_num_of_threads)]

        [thread.start() for thread in threads]

        for ele in self.qa_dataset:
            example = BaseQAData(
                id=ele[ID],
                title=ele[TITLE],
                context=ele[CONTEXT],
                question=ele[QUESTION],
                answer=ele[ANSWER])
            self.input_queue.put(example)

        self.process_event.set()
        [thread.join() for thread in threads]

        json.dump({"new_specials": self.model_utils.special_tokens},
                  open(f"{self.config.pipeline_dataset_folder}/new_special_tokens.json", "w", encoding="utf8"),
                  indent=4, ensure_ascii=False)
        for f in os.listdir(self.output_folder):
            if "all" in f:
                output += load_file(f"{self.output_folder}/{f}")
        self.save_dataset(output)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--constructor_num_of_threads", default=10, type=int)
    # parser.add_argument("--pipeline_pretrained_path", default="vinai/bartpho-word", type=str)
    # parser.add_argument("--pipeline_special_tokens_path", default=SPECIAL_TOKENS_PATH, type=str)
    # parser.add_argument("--pipeline_input_max_length", default=512, type=str)
    # parser.add_argument("--pipeline_dataset_folder", default=DATA_PATH + "/SQuAD_dataset/new_squad/")
    # parser.add_argument("--constructor_ratio", default=[0.9, 0.05, 0.05], type=list)

    # config = parser.parse_args()
    # config = PipelineConfig(**vars(config))

    dataset_construct = DatasetConstructor()
    dataset_construct.run()
