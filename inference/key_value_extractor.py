import requests
import os
import json

from tqdm import tqdm

from common.common_keys import *
from common.config import Config


class AttributeExtractor:
    def __init__(self, mapping_file: str):
        self.product_type2question = json.load(open(mapping_file, "r", encoding="utf8"))
        # self.name2type = json.load(open(mapping_file, "r", encoding="utf8"))

    def product_name2type(self, product_name: str):
        for ele in self.product_type2question.keys():
            if ele.lower() in product_name.lower():
                return ele

        return ""

    def extract_product_type(self, passage: str):
        product_name = self.qa_api(context=passage, question="Tên của sản phẩm là gì?")
        product_name = product_name if product_name else "sản phẩm"
        return self.product_name2type(product_name=product_name), product_name

    def get_questions_from_product_type(self, product_type: str):
        return self.product_type2question.get(product_type, {})

    @staticmethod
    def qa_api(context, question):
        payload = {
            "index": "ascascas",
            "data": {
                CONTEXT: context,
                QUESTION: question
            }
        }

        output = requests.post(url=Config.qa_url, json=payload).json()
        return output["data"][ANSWER]

    @staticmethod
    def create_key2question_dict(input_folder: str, sampling_folder: str):
        sub_folders = os.listdir(input_folder)

        file2type = {}
        type2attrs = {}
        count = 0
        for sub in tqdm(sub_folders):
            file2type_temp = {
                f.replace(".json", ""): sub
                for f in os.listdir(f"{input_folder}/{sub}/text")
            }
            file2type = {**file2type, **file2type_temp}

        sampling_files = os.listdir(sampling_folder)
        for f in tqdm(sampling_files):
            data = json.load(open(f"{sampling_folder}/{f}", "r", encoding="utf8"))
            # type2attrs[file2type[f.replace(".json", "")]] = data[0]["Original"].keys()
            attr2question = {
                "tên sản phẩm": "Tên của sản phẩm là gì?"
            }
            original_reverse = {
                v: k
                for k, v in data[0]["Original"].items()
            }
            for example in tqdm(data):
                if not example[ANSWER]:
                    continue

                try:
                    attr2question[original_reverse[example[ANSWER]]] = example[PREDICTED_QUESTION][0]
                except:
                    count += 1
                    pass

            type2attrs[file2type[f.replace(".json", "")]] = attr2question

        print(count)
        json.dump(type2attrs, open(f"type2attrs.json", "w", encoding="utf8"), ensure_ascii=False, indent=4)

    def extract(self, passage: str):
        product_type, product_name_in_passage = self.extract_product_type(passage=passage)
        key_to_questions = self.get_questions_from_product_type(product_type=product_type)
        attributes = {}

        for key, question in key_to_questions.items():
            question = question.replace("_", " ")
            product_name_in_question = self.qa_api(context=question, question="Tên của sản phẩm là gì?")
            product_name_in_question = product_name_in_question if product_name_in_question else "sản phẩm"

            new_question = question.replace(product_name_in_question, product_name_in_passage)
            value_of_attr = self.qa_api(context=passage, question=new_question)
            if value_of_attr:
                attributes[key] = [value_of_attr, new_question]

        return attributes


if __name__ == "__main__":
    pass
