import argparse
import json
import random

import regex as re
import pandas as pd
from tqdm import tqdm

from common.common_keys import *
from common.constants import *
from common.config import SingletonMeta, QuestionType, ModelInputTag, SamplingType, PipelineConfig
from common.utils import pre_process
from inference.sampling_pipeline import QuestionSampler
from trainer.model.bartpho import BartPhoPointer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


def training_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bartpho_restore_folder",
                        default=CURRENT_PATH + "/inference/checkpoint/bartpho_nopointer_108500steps",
                        type=str)
    parser.add_argument("--bartpho_pointer_restore_folder",
                        default=OUTPUT_PATH + "/checkpoint/checkpoint1_8_paper_attn/checkpoint-24000",
                        type=str)
    parser.add_argument('--sampling_type', default="shopee", type=str,
                        help='dataset name to sampling')
    parser.add_argument('--return_entity', action='store_true',
                        help='whether or not return entity in sampling output')
    parser.add_argument('--verify', action='store_true',
                        help='whether or not verify output question')
    parser.add_argument('--need_sampling', action='store_true',
                        help='whether need sampling or not')
    parser.add_argument('--output_folder', default=SAMPLING_FOLDER + "/wiki_sampling/samplings_26_10", type=str)
    parser.add_argument('--model_device', default="cpu", type=str)

    parser.add_argument("--newest_pointer",
                        default=OUTPUT_PATH + "/checkpoint/checkpoint_bart_20_9/checkpoint-2000",
                        type=str)
    parser.add_argument("--folder_checkpoint", default=INFERENCE_PATH + "/checkpoint/bartpho_pointer_22_9/", type=str)
    parser.add_argument('--input_max_length', default=512, type=int,
                        help='maximum context token number')
    parser.add_argument('--output_max_length', default=256, type=int,
                        help='maximum context token number')
    parser.add_argument('--parallel_input_processing', action='store_true')
    parser.add_argument('--inference_batch_size', default=4, type=int)

    return parser.parse_args()


class QASampling(metaclass=SingletonMeta):
    def __init__(self, config: PipelineConfig):
        self.sampling_type = SamplingType(config.sampling_type)
        self.TYPE_TO_SAMPLING_FUNC = {
            SamplingType.SHOPEE: self.shoppe_sampling,
            SamplingType.WIKI: self.wiki_sampling,
            SamplingType.TGDD: self.tgdd_sampling,
            SamplingType.SQUAD: self.SQuAD_sampling,
            SamplingType.TINHTE: self.tinhte_sampling,
        }
        self.output_folder = config.sampling_dataset_folder
        if True:
            self.sampler = QuestionSampler(config)
        else:
            self.sampler = None

    @pre_process
    def qa_sampling(self, passage: str, output_file: str, _id: str = None):
        samplings = self.sampler.sampling(passage=passage, _id=_id)
        if not samplings:
            print(f"SAMPLING OUTPUT OF {_id} IS EMPTY!!!")

        json.dump(samplings, open(f"{self.output_folder}/{output_file}", "w", encoding="utf8"), ensure_ascii=False,
                  indent=4)

    def get_input_sampling_folder(self, sampling_type: SamplingType):
        input_folder = SAMPLING_FOLDER + f"/{sampling_type.value}_sampling/original/"
        if self.sampling_type.value == SamplingType.TGDD.value:
            input_folder = SAMPLING_FOLDER + f"/{sampling_type.value}_sampling/processed_data/"
        return input_folder

    def common_sampling(self):
        input_folder = self.get_input_sampling_folder(sampling_type=self.sampling_type)
        sampler = self.TYPE_TO_SAMPLING_FUNC[self.sampling_type]
        sampler(input_folder=input_folder)

    @staticmethod
    def convert_QGexample_to_SQuADexample(example: dict, index: int):
        """convert sampling example to SQuAD example for QA training

        Args:
            example (dict): form {
                MODEL_INPUT : input passage to QG model, contain CLUE tag, question type tag and answer
                ANSWER : answer,
                "ques_type" : type of question that generated question follow
                MODEL_QUESTION_TYPE_INPUT: question that predicted from QG model

            }
            index (int): index to save example

        Returns:
            dict: example with form of QA dataset
        """
        process_passage = example[MODEL_INPUT].replace("_", " ").replace(f"{ModelInputTag.clue} ", "").replace(
            f" {ModelInputTag.close_clue}",
            "").replace("<{}> ".format(example[MODEL_QUESTION_TYPE_INPUT].replace("_", " ")), "")
        answer = example[ANSWER].replace("_", " ").strip()
        ans_tag = re.findall(r"<.*?>", process_passage)
        if not ans_tag or len(ans_tag) != 2:
            return None
        start_idx = process_passage.find(f"{ans_tag[0]} {answer} {ans_tag[1]}")
        if start_idx == -1:
            return None
        return {
            "id": str(index),
            "title": str(1000000 + index),
            CONTEXT: process_passage.replace(ans_tag[0] + " ", "").replace(" " + ans_tag[1], ""),
            "answers": {
                "answer_start": [
                    start_idx
                ],
                "text": [
                    answer
                ]
            },
            QUESTION: example[MODEL_LABEL][0].replace("_", " ")
        }

    def convert_samplings_to_SQuAD(self, input_folder: str, output_file: str):
        """convert all sampling dataset to QA dataset

        Args:
            input_folder (str): folder that contain many .json file
            output_file (str): _description_
        """
        assert os.path.isdir(input_folder)
        all_files = os.listdir(input_folder)
        all_data = []
        for ele in tqdm(all_files):
            all_data += json.load(open(f"{input_folder}/{ele}", "r", encoding="utf8"))
        print(all_data[-1])
        print(f"Loaded {len(all_data)} examples")
        squad_dataset = []

        print("Converting ... ")
        for idx, ele in enumerate(tqdm(all_data)):
            d = self.convert_QGexample_to_SQuADexample(ele, idx)
            if d:
                squad_dataset.append(d)
        print(f"Converted {len(squad_dataset)} examples")
        json.dump({"data": squad_dataset}, open(output_file, "w", encoding="utf8"), ensure_ascii=False, indent=4)

    def SQuAD_sampling(self, input_folder: str):
        dataset = []
        # input_folder = "/TMTAI/KBQA/minhbtc/ACS-QG/QASystem/TextualQA/QuestionAnswering_Generation/dataset/SQuAD_dataset/new_squad"
        for f in os.listdir(input_folder):
            if "train" in f or "ML4U" in f:
                dataset += json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))["data"]

        dataset = [ele[CONTEXT] for ele in dataset]
        dataset = list(set(dataset))
        print(f"Loaded {len(dataset)} examples ...")

        count = 410
        init_idx = 410
        for idx, ele in enumerate(tqdm(iterable=dataset)):
            if idx < init_idx:
                continue

            self.qa_sampling(passage=ele, output_file=f"{count}.json")
            count += 1

    def wiki_sampling(self, input_folder: str):
        with open(SAMPLING_FOLDER + "/wiki_sampling/wiki_examples.txt", "r", encoding="utf8") as f:
            done_datas = f.readlines()
        done_datas = [e[:-1] for e in done_datas]
        print("Done data:", len(done_datas))

        def preprocess_passage(sentence):
            sentence = re.sub(r"<!--.*?-->", "", sentence)
            sentence = re.sub(r"\.+", ".", sentence)
            sentence = re.sub(r"([,\.:;])\s?(\.)", r"\1", sentence)
            sentence = re.sub(
                r"([a-z|{}|\d])(\.)([A-Z|{}])".format(VIETNAMESE_RE.lower(), VIETNAMESE_RE),
                r"\1\2 \3", sentence)
            sentence = re.sub(r" Dữ liệu liên quan tới.*|Phương tiện liên quan tới.*", "", sentence).replace("\xa0", "")
            sentence = re.split("\n+", sentence)
            sentence = [ele.strip() for ele in sentence if len(ele) > 100 and not len(ele) == len(ele.encode())]
            return " ".join(sentence)

        all_files = os.listdir(input_folder)
        for ii, f in enumerate(tqdm(all_files)):
            if f.replace(".txt", "") in done_datas:
                continue

            with open(f"{input_folder}/{f}", "r", encoding="utf8") as ff:
                data = ff.read().replace('\n', '.')
            passage = preprocess_passage(data)
            if len(passage) > 10000:
                continue
            self.qa_sampling(passage=passage, output_file=f.replace(".txt", ".json"), _id=f)

    @staticmethod
    def load_shopee_data(input_folder: str):
        all_files = os.listdir(input_folder)
        output = []
        for f in tqdm(all_files):
            data = json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))
            if not isinstance(data, list):
                continue

            for idx, ele in enumerate(data):
                product_name = ""
                additional_details = {
                    k.lower(): v
                    for k, v in ele["new details"].items() if
                    len(k.split()) <= 5 and not k.lower().startswith("thông tin sản phẩm")
                }
                shop_infor = ele["Shop information"]
                if shop_infor and "Thông tin chi tiết" in shop_infor:
                    shop_infor = {
                        k.lower(): v
                        for k, v in ele["Shop information"]["Thông tin chi tiết"].items()
                    }
                else:
                    shop_infor = {}

                main_infor = ele["Main information"]
                if not main_infor:
                    main_infor = {}
                main_infor_dict = {}
                for k, v in main_infor.items():
                    if k == "Name":
                        main_infor_dict["tên sản phẩm"] = v.replace("Yêu Thích. ", "").replace("Yêu Thích . ",
                                                                                               "") if isinstance(v,
                                                                                                                 str) else " ".join(
                            v)
                        product_name = main_infor_dict["tên sản phẩm"]
                    elif k == "Price":
                        for l, ll in main_infor[k].items():
                            if ll:
                                main_infor_dict[l.lower()] = ll
                    elif isinstance(v, list):
                        main_infor_dict[k.lower()] = ", ".join(v)

                detail_infor = {
                    k.lower(): v
                    for k, v in ele["Detail"].items() if k.lower() != "danh mục"
                }
                all_infor = {
                    k: v[0] + v[1:].lower() if isinstance(v, str) and v.isupper() else v
                    for k, v in {
                        **main_infor_dict,
                        **shop_infor,
                        **detail_infor,
                        **additional_details
                    }.items()
                }
                st = []
                value_lst = []
                for k, v in all_infor.items():
                    try:
                        st.append(k + " " + v + ". ")
                        value_lst.append(v)
                    except:
                        continue
                output.append({
                    "id": f.replace(".json", "") + str(idx),
                    "product_name": product_name,
                    CONTEXT: st,
                    "value_lst": value_lst,
                    "original": all_infor
                })
        json.dump(output, open("shopee_processed_data.json", "w", encoding="utf8"), indent=4, ensure_ascii=False)

        return output

    def shoppe_sampling(self, input_folder: str):
        """sampling dataset from shopee key-value data

        Args:
            input_folder (str): --
            input_folder (_type_): contain many examples with form
                    {
                        product_name: <name of product>,
                        context: <list of sentence, each of them contain key and value of attribute of product>,
                        value_lst: <list of value of attribute correspond to each sentence in context>
                    }

        Return: --
        """
        input_dataset = self.load_shopee_data(input_folder=input_folder)

        exist_file = os.listdir(self.output_folder)
        exist_file = [e.replace(".json", "") for e in exist_file]
        print(len(exist_file))
        random.shuffle(input_dataset)
        for example in tqdm(input_dataset):
            _id = example["id"]
            if _id in exist_file:
                continue
            passage_lst = []
            original = []
            answer_lst = []
            for idx, _ in enumerate(example[CONTEXT]):
                # passage = f"<{QuestionType.WHAT.name}> " + "".join([f"{ModelInputTag.clue} " + ele.replace(
                #     example["value_lst"][idx], f"{ModelInputTag.answer} " + example["value_lst"][
                #         idx] + f" {ModelInputTag.close_answer}") + f" {ModelInputTag.close_clue} . " if i == idx else ele
                #                                                     for i, ele in enumerate(example[CONTEXT])])
                #
                # passage = " ".join(
                #     [" ".join([e for e in ele]).replace("< ", "<").replace(" >", ">").replace("/ ", "/") for ele in
                #      Config.vncore_nlp.tokenize(passage)])
                # passage = self.sampler.model_utils.truncate_passage(passage_text=passage)
                passage = "".join([ele.replace(example["value_lst"][idx],
                                               f"{ModelInputTag.answer} " + example["value_lst"][
                                                   idx] + f" {ModelInputTag.close_answer}")
                                   if i == idx else ele for i, ele in enumerate(example[CONTEXT])])
                passage = self.sampler.pre_process_input(passage_ans_clue=passage, question_type=QuestionType.WHAT.name)
                passage_lst.append(passage)
                answer_lst.append(example["value_lst"][idx])
                original.append(example["original"])

            if not passage_lst:
                continue
            output = self.sampler.base_sampling(_id=_id, passage_lst=passage_lst, answer_lst=answer_lst,
                                                original=original,
                                                ques_type_lst=[QuestionType.WHAT.name] * len(passage_lst))
            json.dump(output, open(f"{self.output_folder}/{_id}.json", "w", encoding="utf8"), indent=4,
                      ensure_ascii=False)

    @staticmethod
    def gen_QG_dataset_from_partime(input_folder: str, data_type: str):
        """
        Args:
            input_folder (str): contain many .xlsx file of data <>
            data_type (str): --
        """

        def convert_dict_to_passage(d: dict, key: str):
            p = ""
            for k, v in d.items():
                if k == key:
                    p += f"{ModelInputTag.clue} {k} {ModelInputTag.answer} {v} {ModelInputTag.close_answer} {ModelInputTag.close_clue} . "
                else:
                    p += k + " " + v + ". "

            return p.strip()

        def get_entity_dict_n_passage(type_data: str, folder: str, file_name: str, id_examples: str):
            """
            Get entity dict from raw data

            :param folder:
            :param type_data:
            :param file_name: name of labeled data file
            :param id_examples: id of example to get entity dict
            :return:
            """
            prefix_raw_filename = r"dataset_"
            prefix_labeled_filename = f"{type_data}_dataset_"
            raw_filename = file_name.replace(prefix_labeled_filename, prefix_raw_filename)
            raw_data = pd.read_excel(f"{folder}/{raw_filename}", engine="openpyxl")

            row = raw_data[raw_data["id"] == id_examples]
            entity_dict = row["entity"][0] if "entity" in raw_data.columns and len(row["entity"]) > 0 else {}
            passage = row[PASSAGE] if PASSAGE in raw_data.columns and len(row[PASSAGE]) > 0 else ""
            return entity_dict, passage

        assert os.path.isdir(input_folder)
        output = []
        all_files = os.listdir(input_folder)
        for f in tqdm(all_files):
            data = pd.read_excel(f"{input_folder}/{f}", engine="openpyxl")
            check_column_name = [e for e in data.columns.values if e.startswith("True/False")][0]
            true_ques_column_name = [e for e in data.columns.values if e.startswith("True Question")][0]

            for i in range(data.shape[0]):
                ques_type = QuestionType.WHAT.name if "question type" not in data.columns.values else data.iloc[i][
                    "question type"]
                if pd.isna(data.iloc[i][check_column_name]) or pd.isna(data.iloc[i][true_ques_column_name]):
                    continue

                entity_dict = {}
                answer = data.iloc[i][ANSWER]
                if data_type == SamplingType.SHOPEE.value:
                    passage = convert_dict_to_passage(d=json.loads(data.iloc[i]["Original"]), key=data.iloc[i]["Key"])
                elif data_type == SamplingType.TGDD.value:
                    passage, _ = get_entity_dict_n_passage(type_data=data_type,
                                                           folder=input_folder.replace("raw", "labeled"),
                                                           file_name=f, id_examples=data.iloc[i]["id"])
                elif data_type == SamplingType.TINHTE.value:
                    passage, entity_dict = get_entity_dict_n_passage(type_data=data_type,
                                                                     folder=input_folder.replace("raw", "labeled"),
                                                                     file_name=f, id_examples=data.iloc[i]["id"])
                else:
                    print(data_type, SamplingType.SHOPEE.name, data_type == SamplingType.SHOPEE.name)
                    passage = data.iloc[i][PASSAGE]
                output.append({
                    MODEL_INPUT: passage,
                    MODEL_LABEL: data.iloc[i][true_ques_column_name],
                    MODEL_ENTITY_DICT_INPUT: entity_dict,
                    ANSWER: answer,
                    MODEL_QUESTION_TYPE_INPUT: ques_type
                })
                if not pd.isna(data.iloc[i][check_column_name]) and int(data.iloc[i][check_column_name]) \
                        and data.iloc[i]["predicted question"] != data.iloc[i][true_ques_column_name]:
                    output.append({
                        MODEL_INPUT: passage,
                        MODEL_LABEL: data.iloc[i]["predicted question"],
                        MODEL_ENTITY_DICT_INPUT: entity_dict,
                        ANSWER: answer,
                        MODEL_QUESTION_TYPE_INPUT: ques_type
                    })

        return output

    def tinhte_sampling(self, input_folder: str):
        all_files = os.listdir(input_folder)
        for f in tqdm(all_files):
            try:
                data = json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))
            except json.decoder.JSONDecodeError:
                continue

            if not ("content" in data.keys() and data["content"]):
                continue
            passage = " ".join(data["content"]).replace("\n", "")

            self.qa_sampling(passage=passage, output_file=f)

    @staticmethod
    def load_tgdd_dataset(input_folder: str = None, output_folder: str = None):
        sub_folders = os.listdir(input_folder)

        for sub in tqdm(sub_folders):
            all_files = os.listdir(f"{input_folder}/{sub}/text")
            for f in tqdm(all_files[:10]):
                output = []
                data = json.load(open(f"{input_folder}/{sub}/text/{f}", "r", encoding="utf8"))
                id_ = data[EXAMPLE_ID]
                if not data["technical_information"]:
                    continue

                passage = ["Sản phẩm " + data["main_information"]["name"] + "."]
                answer = [""]
                original = {}
                if isinstance(data["technical_information"], list):
                    inf = data["technical_information"]
                else:
                    inf = [data["technical_information"]]
                for tech in inf:
                    # name_element = tech["name_element"]
                    for ele in tech["attr"]:
                        passage += [
                            e[:-1] + f" là {ModelInputTag.answer} " + ", ".join(k) + f" {ModelInputTag.close_answer}."
                            for e, k in ele.items()]
                        answer += [", ".join(k) for e, k in ele.items()]
                        original = {**original, **ele}
                output.append({
                    EXAMPLE_ID: id_,
                    PASSAGE: passage,
                    ORIGINAL: original,
                    ANSWER: answer
                })
                json.dump(output, open(f"{output_folder}/{f}", "w", encoding="utf8"), ensure_ascii=False, indent=4)

    def tgdd_sampling(self, input_folder: str):
        input_files = os.listdir(input_folder)
        input_dataset = []
        for f in tqdm(input_files):
            input_dataset += json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))

        exist_file = os.listdir(self.output_folder)
        exist_file = [e.replace(".json", "") for e in exist_file]

        for example in tqdm(input_dataset):
            id_ = example[EXAMPLE_ID]
            if id_ in exist_file:
                continue
            passage_lst = []
            original = []
            answer_lst = []
            for idx, p in enumerate(example[PASSAGE]):
                # passage = "<{}> ".format(QuestionType.WHAT.name) + " ".join(
                #     [f"{ModelInputTag.clue} {p} {ModelInputTag.close_clue} ." if i == idx else ele.replace(
                #         f"{ModelInputTag.answer} ", "").replace(f" {ModelInputTag.close_answer}", "") \
                #      for i, ele in enumerate(example[PASSAGE])])
                #
                # passage = " ".join(
                #     [" ".join([e for e in ele]).replace("< ", "<").replace(" >", ">").replace("/ ", "/") for ele in
                #      Config.vncore_nlp.tokenize(passage)])
                # passage = self.sampler.model_utils.truncate_passage(passage_text=passage)
                passage = " ".join([p if i == idx else ele.replace(f"{ModelInputTag.answer} ", "").replace(
                    f" {ModelInputTag.close_answer}", "") for i, ele in enumerate(example[PASSAGE])])
                passage = self.sampler.pre_process_input(passage_ans_clue=passage, question_type=QuestionType.WHAT.name)
                passage = passage.replace(" . .", " .")
                passage_lst.append(passage)
                answer_lst.append(example[ANSWER][idx])
                original.append({
                    k: ", ".join(v)
                    for k, v in example["original"].items()
                })
            if not passage_lst:
                continue

            output = self.sampler.base_sampling(_id=id_, passage_lst=passage_lst, answer_lst=answer_lst,
                                                original=original,
                                                ques_type_lst=[QuestionType.WHAT.name] * len(passage_lst))
            json.dump(output, open(f"{self.output_folder}/{id_}.json", "w", encoding="utf8"), indent=4,
                      ensure_ascii=False)


if __name__ == "__main__":
    config = training_config()
    sampling = QASampling(config, need_sampling=True)
    sampling.common_sampling()
    from common.constants import *
