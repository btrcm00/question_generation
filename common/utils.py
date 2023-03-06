import pickle
import sys
import time
import copy
import numpy as np
import requests
import string

import regex as re
from googletrans import Translator
from nltk import tokenize as nltk_tokenizer
from queue import Queue
from threading import Event

from common.common_keys import *
from common.config import *
from common.constants import *

translator = Translator()


def tone_normalization(passage):
    for i, j in Config.tone_mapping.items():
        passage = passage.replace(i, j)
    return passage


def post_process(func):
    def _post_process(*args, **kwargs):
        passage = func(*args, **kwargs)

        passage = re.sub(r"(\. )+", ". ", passage)
        passage = re.sub(r"( \.)+", " .", passage)
        passage = tone_normalization(passage)
        return passage

    return _post_process


def pre_process(func):
    def _pre_process(*args, **kwargs):
        passage = kwargs.get("passage")

        passage = tone_normalization(passage)
        # passage = passage.replace("_", " ").replace("\"", "'")
        passage = re.sub(r"\s\"\s", r" ", passage)
        passage = re.sub(r"(\w)([\.,:;])(\w)", r"\1\2 \3", passage)
        # passage = re.sub(r"([\.,;]) ([A-Z]|[{}])".format(VIETNAMESE_RE), r". \2",
        #                  passage)
        kwargs["passage"] = passage
        return func(*args, **kwargs)

    return _pre_process


def translate(sentence: str, src="vi", dest="en"):
    """Translate sentence from vi to en

    Args:
        sentence (str): _description_
        src (str): _description_
        dest (str): _description_
    """
    count = 0
    out = ""
    while count < 5:
        try:
            out = translator.translate(sentence, src=src, dest=dest).text
            break
        except:
            count += 1
    return out


@pre_process
def get_question_style(passage: str):
    """ Classify style of question

    Args:
        passage (str): _description_
    """

    question = translate(passage)

    if "bao nhiêu" in passage:
        return "HOW MANY"
    for ques_type in QuestionType:
        if ques_type.name != "OTHER" and question.upper().startswith(ques_type.name.replace("_", " ")):
            return ques_type.name

    return "OTHER"


def _pickle_dump_large_file(obj, filepath):
    """
    This is a defensive way to write pickle.write,
    allowing for very large files on all platforms
    """
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

    print(f"Save {filepath}-{len(obj)} done!")


def _pickle_load_large_file(filepath):
    """
    This is a defensive way to write pickle.load,
    allowing for very large files on all platforms
    """
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj


def make_request(api_url, data, method):
    count = 0
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(data)
    while count < 10:
        try:
            output = requests.request(method=method, url=api_url, data=payload, headers=headers).json()
            return output
        except:
            count += 1
    return {}


def check_exist_file(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, "w"):
            pass


def save_file(obj, path: str):
    check_exist_file(path)
    _pickle_dump_large_file(obj, filepath=path)


def load_file(path: str):
    return _pickle_load_large_file(filepath=path)


def remove_stop_word(sentence: str):
    sentence = sentence.replace("_", " ")
    sent_processed = [" ".join(ele.split("_")) for ele in [i for j in Config.vncore_nlp.tokenize(sentence) for i in j]
                      if
                      ele not in Config.stop_word_lst and ele not in string.punctuation]

    return " ".join(sent_processed).split()


def check_exist_folder(folder_name: str):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)


def timer(func):
    def _timer(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"`{func.__qualname__}` PROCESSING TIME:", time.time() - start_time)
        return result

    return _timer


class ModelUtils(metaclass=SingletonMeta):
    def __init__(
            self,
            input_max_length: int = None,
            tokenizer=None
    ):
        self.input_max_length = input_max_length
        special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[SPECIAL_TOKENS] + [e.name for e in QuestionType]
        self.special_tokens = list(map(lambda x: x.upper(), special_tokens))
        self.special_pattern = re.compile("<(" + "|".join(special_tokens) + ")>")
        self.tokenizer = tokenizer

    @post_process
    def truncate_passage(self, passage: str):
        """truncate passage to len(tokens of truncated passage) < max_length of model
        Condition: input passage have to be segmented (by VnCoreNLP).

        Args:
            passage (str): _description_

        Returns:
            str: truncated passage
        """
        passage_text = passage.replace(f". {ModelInputTag.close_clue}", f"{ModelInputTag.close_clue} .").replace(
            "\xa0", "")
        passage_text = re.sub(r"([a-z])(\.)([A-Z])", r"\1 \2 \3", passage_text)

        if len(self.tokenizer(passage_text)[INPUT_IDS]) <= self.input_max_length:
            return passage_text

        passage_splits = [ele + " ." for ele in passage_text.split(" . ")]
        out_passage = copy.deepcopy(passage_splits)
        while len(self.tokenizer(" ".join([e for ele1 in out_passage for e in ele1.split()]))[
                      INPUT_IDS]) > self.input_max_length:
            retain_idx = [idx for idx, ele in enumerate(out_passage) if not self.special_pattern.search(ele)]
            if not retain_idx:
                print(out_passage)
                print(len([e for ele1 in out_passage for e in ele1.split()]))
                break
            rm_index = random.choice(retain_idx)
            out_passage.pop(rm_index)

        return " ".join(out_passage)

    @post_process
    def prepare_model_input(self, passage: str, answer: str, ans_lst: list, ques_type: str, ans_type: str = None, ):
        """Add CLUE tag and answer tag to passage

        Args:
            passage (str): _description_
            answer (str): _description_
            ans_lst (list): _description_
            ans_type (str, optional): _description_. Defaults to None.
            ques_type

        Returns:
            prepared passage: _description_
        """

        if ans_type is None:
            ans_type = ans_lst[0]
        answer_chunk = f"<{ans_type}> {answer} </{ans_type}>"
        passage = passage[:ans_lst[1]] + answer_chunk + passage[ans_lst[2]:]
        split_passage = passage.split(" . ")
        passage_ans_clue = " ".join(
            f"{ModelInputTag.clue} {ele} {ModelInputTag.close_clue} ." if answer_chunk in ele else f"{ele} ." for ele in
            split_passage)
        passage_ans_clue = self.truncate_passage(f"<{ques_type}> {passage_ans_clue}")
        return passage_ans_clue

    def prepare_model_input_threading(self, bar, output: list, q: Queue, e: Event):
        """Add CLUE tag and answer tag to passage

        Args:
            bar (str): _description_
            output (str): _description_
            q (list): _description_
            e (str, optional): _description_. Defaults to None.

        Returns:
            prepared passage: _description_
        """
        while not e.is_set() or not q.empty():
            data: dict = q.get()
            ans_lst = data.get("ans_lst")
            answer = data.get(ANSWER)
            passage = data.get(PASSAGE)
            ques_type = data.get(QUESTION_TYPE)
            ans_type = data.get("ans_type", ans_lst[0])

            _data = copy.deepcopy(data)

            if ans_type is None:
                ans_type = ans_lst[0]
            answer_chunk = f"<{ans_type}> {answer} </{ans_type}>"
            passage = passage[:ans_lst[1]] + answer_chunk + passage[ans_lst[2]:]
            split_passage = passage.split(" . ")
            passage_ans_clue = " ".join(
                f"{ModelInputTag.clue} {ele} {ModelInputTag.close_clue} ." if answer_chunk in ele else f"{ele} ." for
                ele in split_passage)
            passage_ans_clue = self.truncate_passage(f"<{ques_type}> {passage_ans_clue}")
            _data[PASSAGE] = passage_ans_clue
            output.append(_data)
            bar.update(1)
            print(bar.total, bar.n, bar.n == bar.total)
            if bar.n == bar.total:
                print("ENDDDDDDDDD")
                e.set()
                break

        # return passage_ans_clue

    @staticmethod
    def bert_prepare_data(text: str):
        seg_text = " ".join([e for ele in Config.vncore_nlp.tokenize(text) for e in ele])
        seg_text = seg_text.replace("< ", "<")
        seg_text = seg_text.replace("/ ", "/")
        seg_text = seg_text.replace(" >", ">")
        return seg_text

    def question_validation(self, passage: str, question: str, answer: str, score: int = 0.4):
        """_summary_
        Steps:
            - Entity in question must be in passage
            - Generated question do not contain answer
            - translate question to english => get its question type => check whether or not match with input type

        Args:
            passage (str): _description_
            question (str): _description_
            answer (str): _description_
            score (int)
        Return:
            bool: True if generated question is valid
        """
        ques_type = re.findall(f"<.*?>", passage)
        if ques_type and passage.strip().startswith(ques_type[0]):
            passage = passage.replace("{} ".format(ques_type[0]), "")
        clue = re.findall(r"{} (.*?) {}".format(ModelInputTag.clue, ModelInputTag.close_clue), passage)
        if not clue:
            return False
        passage = passage.replace("{} ".format(ModelInputTag.clue), "").replace(" {}".format(ModelInputTag.close_clue),
                                                                                "")
        answer_truth = re.findall(r"<.*?> (.*?) <.*?>", passage)
        if not answer_truth:
            return False
        else:
            answer_truth = answer_truth[0].replace("_", " ").split()
        passage = re.sub(r"<.*?> ", "", passage)
        # ner_in_question, _ = self.get_entity_from_passage(question.replace("_", " "))
        # if any(ent[0] == answer or ent[0] not in passage \
        #        for ent in ner_in_question.keys()):
        #     return False

        # Overlap tokens ratio between answer and predicted answer from QA model > 0.6
        # try:
        answer_pred = self.qa_api(context=passage, question=question)["data"][ANSWER].replace("_", " ").replace("₫ ",
                                                                                                                "₫").replace(
            " %", "%").split()
        # except:
        #     return False
        # answer_truth = answer.replace("_", " ").split()
        # print(answer_pred, answer_truth)
        if not answer_truth:
            return False
        if sum([1 if e in answer_pred else 0 for e in answer_truth]) / len(answer_truth) < score:
            return False

        # Overlap tokens ratio between question and clue > 0.6
        # not_stop_clue = remove_stop_word(clue.replace("_", " "))
        # not_stop_question = remove_stop_word(question.replace("_", " "))
        # if not not_stop_question:
        #     return False
        # return len([e for e in not_stop_question if e in not_stop_clue]) / len(not_stop_question) > score
        return True

    @pre_process
    def tokenize_passage(self, passage: str, depth: int = 0):
        if depth >= 10:
            return []
        try:
            return Config.vncore_nlp.tokenize(passage)
        except:
            sentences = nltk_tokenizer.sent_tokenize(passage)
            half_l = len(sentences) // 2
            p1 = " ".join(sentences[:half_l])
            p2 = " ".join(sentences[half_l:])
            return self.tokenize_passage(passage=p1, depth=depth + 1) + self.tokenize_passage(passage=p2,
                                                                                              depth=depth + 1)

    @staticmethod
    def _is_linkable_char(text: str):
        if all(char in [",", "và", "với"] for char in text.split()):
            return True
        return False

    def concat_adjacent_entities(self, ner_dict: dict, passage: str):
        output_ner_dict = {}
        ner_list = [(ner_text, ner_lst) for ner_text, ner_lst in ner_dict.items()]
        ner_list.sort(key=lambda ele: ele[1][1])

        temp_lst = []
        for idx in range(len(ner_list)):
            start_pos = ner_list[idx][1][1]
            end_pos = ner_list[idx][1][2]
            tag = ner_list[idx][1][0]
            if temp_lst and tag == temp_lst[0] and (
                    temp_lst[2] + 1 == start_pos or self._is_linkable_char(passage[temp_lst[2]:start_pos])):
                temp_lst = [tag, temp_lst[1], end_pos]
            elif not temp_lst:
                temp_lst = ner_list[idx][1]
            else:
                output_ner_dict[passage[temp_lst[1]: temp_lst[2]]] = temp_lst
                temp_lst = ner_list[idx][1]
        if temp_lst:
            output_ner_dict[passage[temp_lst[1]: temp_lst[2]]] = temp_lst
        return output_ner_dict

    @timer
    def get_entity_from_passage(self, passage, is_segmented_list: bool = False):
        """
        This func to get entities from passage

        if passage is string:
            - call api to get entities as usual
        elif passage is list:
            - passage is list[str]
            - split passage into sub passage and get entity from each sub passage, then concat to get all entity in passage.
            !!! this case to avoid case that passage is too long => ner api will error
        else:
            error

        :param passage:
        :param is_segmented_list: whether passage is segmented or not
        :return:
            entity_dict: dictionary that contain entities and its start and end index
            passage_: passage corresponding to position of entities, use this returned passage to properly get true entities position.
        """
        assert is_segmented_list and isinstance(passage, list), "ERROR!!!!!!!!"
        ner_dict = {}
        if is_segmented_list:
            processed_passage = []
            temp_p = ""
            for p in passage:
                temp_p += p + " "
                if len(temp_p.split()) > 500:
                    processed_passage.append(temp_p)
                    temp_p = ""
            if temp_p:
                processed_passage.append(temp_p)
        else:
            processed_passage = [[0, passage]]

        out_passage = ""
        count = 0
        for sub_passage in processed_passage:
            count += 1
            # output = requests.post(url=Config.ner_url,
            #                        json={"text": sub_passage.replace("_", " "), "keep_format": True}).json()
            output = make_request(api_url=Config.ner_url,
                                  data={"text": sub_passage.replace("_", " "), "keep_format": True}, method="POST")
            if output["metadata"]["status"] == 500 and not output["data"]["tags"]:
                sub_ner_dict = {}
            else:
                sub_ner_dict = {
                    tag["text"].replace(" ", "_") if tag["text"].replace(" ", "_") in sub_passage else tag["text"]: [
                        tag["label"],
                        tag["begin"] + len(out_passage),
                        tag["end"] + len(out_passage)]
                    for tag in output["data"]["tags"]}

                # remove WHO in ner_dict if processed_passage not contain WHO word because WHO is also question type.
                if "WHO" in sub_ner_dict.keys() and "<WHO>" in sub_ner_dict and " WHO" not in sub_ner_dict:
                    sub_ner_dict.pop("WHO")

                out_passage += output["data"]["text"].strip() + " "

            ner_dict = {**ner_dict, **sub_ner_dict}
        ner_dict = self.concat_adjacent_entities(ner_dict, passage=out_passage.strip())

        return ner_dict, out_passage.strip()

    @staticmethod
    def qa_api(context: str, question: str):
        payload = {
            "index": "question_generation",
            "data": {
                CONTEXT: context.replace("_", " "),
                QUESTION: question.replace("_", " ")
            }
        }

        # output = requests.post(url=Config.qa_url, json=payload).json()
        output = make_request(api_url=Config.qa_url, data=payload, method="POST")
        return output

    def _navigate(self, node: dict, tag_lst: list = None):
        """recursive function to get all chunk in parsing tree

        Args:
            node (dict): _description_

        Returns:
            list: _description_
        """
        if tag_lst is None:
            tag_lst = ["NP", "VP", "QP"]
        sents = []
        chunk_list = []

        child_lst = sorted(node["children"], key=lambda t: t["id"])

        for ele in child_lst:
            child_list, child_chunk = self._navigate(ele, tag_lst)
            chunk_list += child_chunk
            sents += child_list

        cur_child = node["form"]
        if cur_child:
            cur_child += "___" + str(node["id"])

        tag = node["phrase_level_tag"] + "-" + node["functional_tag"]
        if any(tag.upper().startswith(t) for t in tag_lst):
            chunk_list += [sents + [tag]]

        if cur_child:
            sents += [cur_child]

        return sents, chunk_list

    @staticmethod
    def parse_sentence(passage: str):
        """get parsing tree of sentence

        Args:
            passage (str): _description_

        Returns:
            dict: _description_
        """
        passage = tone_normalization(passage)
        payload = {
            "text": passage,
            "outputFormat": "dict"
        }
        # try:
        #     parsing_tree = requests.post(Config.parsing_url, data=json.dumps(payload), timeout=5).json()
        # except:
        #     return "ERROR"
        parsing_tree = make_request(api_url=Config.parsing_url, data=payload, method="POST")
        if parsing_tree == "Server error":
            return "ERROR"

        return parsing_tree["ROOT"]

    def get_chunk(self, passage, tag_lst: list = None, is_segmented: bool = False):
        """return chunks of sentence

        Args:
            passage (): __
            tag_lst (list)
            is_segmented (bool)

        Returns:
            list: list of chunks in sentence. [NER_tag, POS_tag, chunk, start_position, end_position]
        """
        # split passage into sentences
        if not tag_lst:
            tag_lst = ["AP", "PP-MNR"]
        if is_segmented:
            sentence_list = [" ".join(ele) for ele in passage]
        else:
            if isinstance(passage, list):
                passage = " ".join(passage).replace("_", " ")
            sentence_list = [" ".join([e for e in ele]) for ele in Config.vncore_nlp.tokenize(passage)]

        index_sentence_in_passage = np.concatenate(
            [[0], np.cumsum([len(ele.split()) for idx, ele in enumerate(sentence_list)])])

        def get_sentence_chunk(sent, start_idx: int):
            try:
                tree = self.parse_sentence(passage=sent)
            except:
                return []

            if not isinstance(tree, dict):
                return []
            chunk_list = []
            _, orig_chunk_list = self._navigate(tree, tag_lst=tag_lst)
            for ele in orig_chunk_list:
                try:
                    chunk = [re.sub(r"___\d+", "", e) for e in ele[:-1]]

                    start_pos = int(ele[0].split("___")[-1]) - 1
                    # start_pos = len(" ".join(sent.split()[:start_pos])) + 1
                    end_pos = int(ele[-2].split("___")[-1]) - 1
                    # end_pos = start_pos + len(" ".join(chunk))
                    chunk_list.append((None, ele[-1], chunk, int(start_pos + start_idx),
                                       int(end_pos + start_idx)))
                except Exception as e:
                    print('Exception in Chunking: ', e)
                    continue

            return chunk_list

        return [e for sent, start_idx in zip(sentence_list, index_sentence_in_passage) for e in
                get_sentence_chunk(sent, start_idx)], " ".join(sentence_list).split()


if __name__ == "__main__":
    pass
    # print(nltk_tokenizer.tokenize("₫270.000"))
