import json
import unicodedata
import regex as re
import string
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from common.constants import TONE_MAPPING_PATH
from common.config import Config, SingletonMeta


# from trainer.model.bartpho import BartPhoPointer


class Normalizer(metaclass=SingletonMeta):
    def __init__(self):
        self.nlp = Config.vncore_nlp
        self.tone_mapping_dict = json.load(open(TONE_MAPPING_PATH, "r", encoding="utf8"))

        self.hash_tag_pattern = re.compile(Config.hashtag_pattern)
        self.html_pattern = re.compile(r"<.*>")
        self.special_pattern = re.compile(r"[\d\w{}]".format(Config.special_pattern))
        self.url_pattern = re.compile(Config.url_pattern)
        self.email_pattern = re.compile(Config.email_pattern)

    def remove_special_tokens(self, sentence: str):
        sentence = self.hash_tag_pattern.sub(" ", sentence)
        sentence = self.html_pattern.sub("", sentence)
        tokens = [" ".join(word.split("_")) for sent in self.nlp.tokenize(sentence) for word in sent]
        filtered_tokens = []
        for idx, token in enumerate(tokens):
            if len(token) == 1 and not self.special_pattern.search(token) and \
                    not (token == "-" and 0 < idx < len(tokens) - 1 and tokens[idx - 1].isdigit() and tokens[
                        idx + 1].isdigit()):
                continue

            if not self.url_pattern.search(token) and not self.email_pattern.search(token):
                token = re.sub(r"[^\d\w\s{}]".format(Config.special_pattern), "", token)
            filtered_tokens.append(token)
        return " ".join(filtered_tokens)

    def tone_mapping(self, sentence):
        for key, value in self.tone_mapping_dict.items():
            sentence = sentence.replace(key, value)
        return sentence

    @staticmethod
    def remove_duplicate_chars(sentence: str):
        specials = r"\n{}".format(Config.special_pattern)
        sentence = re.sub(r"([{}])[{}]+".format(specials, specials), r"\1 ", sentence)
        sentence = re.sub(r"\n-", ". ", sentence)
        sentence = re.sub(r"\n+", ". ", sentence)
        sentence = re.sub(r"\.\.+", ". ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"([^\.\w\d\n])([\n\.]+)", r"\1", sentence)
        return sentence

    def normalize(self, sentence: str):
        # normalized_text = sentence.lower()
        normalized_text = unicodedata.normalize("NFC", sentence)
        normalized_text = self.remove_duplicate_chars(normalized_text)
        normalized_text = self.remove_special_tokens(normalized_text)
        normalized_text = self.tone_mapping(normalized_text)
        return normalized_text

    @staticmethod
    def remove_punc(text: str):
        while text.endswith(tuple(string.punctuation + string.whitespace)):
            text = text[:-1]
        while text.startswith(tuple(string.punctuation + string.whitespace)):
            text = text[1:]

        return text

    def extract_key_value(self, sentence: str):
        processed_text = self.normalize(sentence)
        k_v = {}
        for e in re.findall(r"([\w\d\s]+\s*:\s*(?:{}|[^,\.;?\n]+))".format(Config.url_pattern), processed_text):
            t = e[0].split(":")
            k_v[self.remove_punc(t[0])] = self.remove_punc(":".join(t[1:]))

        return k_v

    def extract(self, input_folder: str, output_folder: str):
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        all_files = os.listdir(input_folder)
        for f in tqdm(all_files):
            output = []
            data = json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))
            if not isinstance(data, list):
                continue
            for example in tqdm(data):
                example["new details"] = self.extract_key_value(example["Description"])
                output.append(example)

            json.dump(output, open(f"{output_folder}/{f}", "w", encoding="utf8"), ensure_ascii=False, indent=4)


class ShopeeProcessor:
    def __init__(self, model_path: str):
        self.normalizer = Normalizer()

        # self.qg_model = BartPhoPointer.from_pretrained(model_path, training_config={"use_pointer": True})
        # self.qg_tokenizer = AutoTokenizer.from_pretrained(model_path)

    def process(self, input_folder: str, _type: str):
        all_files = os.listdir(input_folder)
        output1 = []
        for f in tqdm(all_files):
            output = []
            data = json.load(open(f"{input_folder}/{f}", "r", encoding="utf8"))
            if not isinstance(data, list):
                # continue
                data = [data]
            _id = data[-1]["_id"]
            for ele in data:
                if ele is None:
                    continue
                additional_details = {
                    k.lower(): v if isinstance(v, str) else ", ".join(v)
                    for k, v in ele.get("new details", {}).items() if
                    len(k.split()) <= 5 and not k.lower().startswith("thông tin sản phẩm")
                }
                # shop_infor = ele["shop_information"]
                # if shop_infor is not None and "thông_tin_chi_tiết" in shop_infor:
                #     shop_infor = {
                #         k.lower(): v
                #         for k, v in ele["shop_information"]["thông_tin_chi_tiết"].items()
                #     }
                # else:
                #     shop_infor = {}
                shop_infor = {}

                main_infor = ele["main_information"]
                main_infor_dict = {}
                for k, v in main_infor.items():
                    if k == "name":
                        main_infor_dict["tên sản phẩm"] = v if isinstance(v, str) else ", ".join(v)
                    elif isinstance(v, list):
                        main_infor_dict[k.lower()] = ", ".join(v)

                detail = ele.get("detail", {})
                if detail is None:
                    detail = {}
                detail_infor = {
                    k.lower(): v if isinstance(v, str) else ", ".join(v)
                    for k, v in detail.items() if k.lower() != "danh_mục"
                }
                output.append({
                    k: v[0] + v[1:].lower() if v.isupper() else v
                    for k, v in {
                        **main_infor_dict,
                        **shop_infor,
                        **detail_infor,
                        **additional_details
                    }.items()
                })
                st = ""
                for k, v in output[-1].items():
                    st += k + " : " + v + ". "
                output1.append({"_id": _id, "info": self.normalizer.remove_special_tokens(st), "type": "sản phẩm",
                                "categore": _type})
        return output1


if __name__ == "__main__":
    processor = ShopeeProcessor(model_path="")

    input_folder = "/TMTAI/FileShare/Public/Data/shopee/shopee_ver_4"
    for _type in tqdm(os.listdir(input_folder)[-2:]):
        print("======", _type)
        json.dump(processor.process(f"{input_folder}/{_type}/text", _type=_type),
                  open(f"temp_dataset/{_type}.json", "w", encoding="utf8"), indent=4, ensure_ascii=False)