import argparse

from queue import Queue
from tqdm import tqdm
from transformers import AutoTokenizer
from threading import Thread, Event

from common.utils import *
from common.config import *
from common.constants import DATA_PATH, SPECIAL_TOKENS_PATH


class DatasetConstructor(metaclass=SingletonMeta):
    def __init__(self, constructor_config: PipelineConfig, mode):
        # new_special_tokens = json.load(open(constructor_config.special_tokens_path))[
        #                          SPECIAL_TOKENS] + [e.name for e in QuestionType]
        # new_specials_lst = [f"<{tk.upper()}>" for tk in list(set(new_special_tokens))]
        tokenizer = AutoTokenizer.from_pretrained(constructor_config.pipeline_pretrained_path)

        self.model_utils = ModelUtils(input_max_length=constructor_config.pipeline_input_max_length,
                                      tokenizer=tokenizer)

        self.mode = mode
        self.num_of_threads = constructor_config.constructor_num_of_threads

        output_folder = f"{constructor_config.pipeline_dataset_folder}/processed"
        check_exist_folder(output_folder)
        self.output_folder = output_folder

        assert sum(constructor_config.constructor_ratio) == 1, "Sum of ratio must be 1"
        self.ratio_for_split = constructor_config.constructor_ratio

        input_data_folder = f"{constructor_config.pipeline_dataset_folder}/raw"
        assert os.path.isdir(input_data_folder)
        self.qa_dataset = self.load_qa_dataset(data_path=input_data_folder, mode=self.mode)

    @staticmethod
    def load_qa_dataset(data_path, mode="train"):
        """
        Dataset format:
        {
            "data": [
                {
                    "title": <title>,
                    "context": context,
                    "id": "uit_000001",
                    "answers": {
                        "answer_start": [
                            start position of answer in context
                        ],
                        "text": [
                            <answer text>
                        ]
                    }
                },
                ...
            ]
        }
        """
        file_lst = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                    os.path.isfile(os.path.join(data_path, f)) and mode in f]
        dataset = []
        for f in file_lst:
            dataset += json.load(open(f, "r", encoding="utf8"))["data"]
        print(f"Loaded {len(dataset)} examples")
        return dataset

    def process_data(self, bar, output: list, q: Queue, e: Event):
        while not e.is_set() or not q.empty():
            data = q.get()

            if data["answers"]["text"] and data["answers"]["text"][0] in data[CONTEXT]:
                answer = " ".join(
                    [" ".join([e for e in ele]) for ele in Config.vncore_nlp.tokenize(data["answers"]["text"][0])])
                answer = tone_normalization(answer)
                while answer.endswith(tuple(string.punctuation + string.whitespace)):
                    answer = answer[:-1]

                passage_ans_clue = data[CONTEXT]
                ner_dict, processed_passage = self.model_utils.get_entity_from_passage(passage_ans_clue)
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

                    passage_ans_clue = self.model_utils.prepare_model_input(passage=processed_passage, answer=answer,
                                                                            ans_lst=ner_lst)
                    ques_ans_label = self.model_utils.bert_prepare_data(
                        data[QUESTION])  # + f" <{ner_ans}> {answer} </{ner_ans}>")

                    if all(e in passage_ans_clue for e in
                           ["CLUE", "/CLUE", ner_ans, "/" + ner_ans]) and "{hl" not in passage_ans_clue:
                        if passage_ans_clue:
                            output.append({
                                "id": data["id"],
                                ANSWER: answer,
                                MODEL_INPUT: passage_ans_clue,
                                MODEL_LABEL: ques_ans_label,
                                MODEL_QUESTION_TYPE_INPUT: get_question_style(sentence=data[QUESTION]),
                                MODEL_ENTITY_DICT_INPUT: ner_dict
                            })
                    if len(output) % 500 == 0:
                        save_file(output, self.output_folder + f"/{self.mode}_examples.pkl")

            bar.update(1)

    def save_dataset(self, dataset: list, shuffle: bool = True):
        if shuffle:
            random.shuffle(dataset)

        l = len(dataset)
        ratio = [sum(self.ratio_for_split[:i]) for i in range(1, len(self.ratio_for_split))]
        print(ratio)
        train_data = dataset[:int(l * ratio[0])]
        dev_data = dataset[int(l * ratio[0]):int(l * ratio[1])]
        test_data = dataset[int(l * ratio[1]):]

        save_file(train_data, self.output_folder + "/train_examples.pkl")
        save_file(dev_data, self.output_folder + "/dev_examples.pkl")
        save_file(test_data, self.output_folder + "/test_examples.pkl")

    def main(self):
        output = []
        input_queue = Queue(maxsize=1000)
        bar = tqdm(total=len(self.qa_dataset), initial=0, leave=True)

        event = Event()
        event.clear()

        threads = [Thread(target=self.process_data, args=(bar, output, input_queue, event), daemon=True) for _ in
                   range(self.num_of_threads)]

        [thread.start() for thread in threads]

        for ele in self.qa_dataset:
            input_queue.put(ele)

        event.set()
        [thread.join() for thread in threads]

        json.dump({"new_specials": self.model_utils.special_tokens},
                  open("new_special_tokens.json", "w", encoding="utf8"),
                  indent=4, ensure_ascii=False)
        save_file(output, self.output_folder + f"/{self.mode}_examples.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constructor_num_of_threads", default=10, type=int)
    parser.add_argument("--pipeline_pretrained_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--pipeline_special_tokens_path", default=SPECIAL_TOKENS_PATH, type=str)
    parser.add_argument("--pipeline_input_max_length", default=512, type=str)
    parser.add_argument("--pipeline_dataset_folder", default=DATA_PATH + "/SQuAD_dataset/new_squad/")
    parser.add_argument("--constructor_ratio", default=[0.9, 0.05, 0.05], type=list)

    config = parser.parse_args()
    config = PipelineConfig(**vars(config))

    dataset_construct = DatasetConstructor(constructor_config=config, mode="ML4U")
    dataset_construct.main()
