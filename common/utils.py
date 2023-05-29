import pickle
import sys
import time
import copy
import numpy as np
import requests
import string
import logging
import json
import hashlib

import regex as re
from googletrans import Translator
from nltk import tokenize as nltk_tokenizer
from queue import Queue
from threading import Event
from string import punctuation
from typing import Tuple, List

from common.common_keys import *
from common.config import *
from common.constants import *

translator = Translator()
logger = logging.getLogger(__name__)


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


def base_pre_process(passage):
    passage = tone_normalization(passage)
    # passage = passage.replace("_", " ").replace("\"", "'")
    passage = re.sub(r"\s\"\s", r" ", passage)
    passage = re.sub(r"(\s*)(\W)\W+", r"\1\2", passage)
    passage = re.sub(r"([^\d\W])([\.,:;])([^\d\W])", r"\1\2 \3", passage)
    passage = re.sub(r"([^\d\W])([\.,:;])(\d)", r"\1\2 \3", passage)
    passage = re.sub(r"(\d)([\.,:;])([^\d\W])", r"\1\2 \3", passage)
    return passage


def pre_process(func):
    def _pre_process(*args, **kwargs):
        passage = kwargs.get("passage")

        passage = base_pre_process(passage=passage)
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

    logger.info(f"Save {filepath}-{len(obj)} done!")


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
    return requests.request(method=method, url=api_url, data=payload, headers=headers).json()


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
        logger.info(f"`{func.__qualname__}` PROCESSING TIME: {time.time() - start_time}")
        return result

    return _timer


class VietnameseTextNormalizer:
    vowels_pattern = re.compile(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|'
        r'ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|'
        r'ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|'
        r'À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|'
        r'Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
    )
    component_pattern = re.compile(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)')

    def convert_unicode(self, text: str) -> str:
        """
        :param text: input text
        :return: utf-8 encoded text
        """
        character_map = dict()
        char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|" \
                   "ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|" \
                   "ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|" \
                   "À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|" \
                   "Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|" \
                   "Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')

        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|" \
                   "ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|" \
                   "Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')

        for i in range(len(char1252)):
            character_map[char1252[i]] = charutf8[i]

        return self.vowels_pattern.sub(lambda x: character_map[x.group()], text)

    @staticmethod
    def normalize_mark(word: str) -> str:
        """
        Normalize vietnamese mark_idx with modern style
        :param word: input word (must be lower case)
        :return: normalized word
        >>> text_norm = VietnameseTextNormalizer()
        >>> text_norm.normalize_mark('Thuỵ')
        >>> "Thụy"
        """
        vowels = [['a', 'à', 'á', 'ả', 'ã', 'ạ'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']]

        vowel_positions = dict()
        for i in range(len(vowels)):
            for j in range(len(vowels[i])):
                vowel_positions[vowels[i][j]] = (i, j)

        # Check whether word is Vietnamese word or not
        # TODO: The below code idea is not optimal, better fix this or remove
        # Main idea: If distance between two vowels in a word is greater than 2 index
        # Then, this word is not a Vietnamese word
        # Fail cases are non-sense words, e.g hoaua

        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, y = vowel_positions.get(char, (-1, -1))
            if x != -1:  # this character is a consonant
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return word
                    vowel_index = index
        # -----------------------------------------------

        mark_idx = 0
        vowel_indexes = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = vowel_positions.get(char, (-1, -1))
            if x == -1:  # char is a consonant
                continue

            elif x == 9:  # if char is "u"
                if index != 0 and chars[index - 1] in 'q':  # if previous char is q
                    chars[index] = 'u'  # remove mark
                    qu_or_gi = True

            elif x == 5:  # if char is i
                if index != 0 and chars[index - 1] == 'g':  # if previous char is g
                    chars[index] = 'i'  # remove mark
                    qu_or_gi = True

            # Save the position of the main mark in word
            if y != 0:
                mark_idx = y
                chars[index] = vowels[x][0]  # remove mark

            # Save the position of the vowel in word that doesn't start with qu or gi
            if not qu_or_gi or index != 1:
                vowel_indexes.append(index)

        # If the number of vowels in the word is less than 2. E.g: gà, giá, quà
        if len(vowel_indexes) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = vowel_positions.get(chars[1])
                    chars[1] = vowels[x][mark_idx]  # the mark is in the last letter, E.g: gà
                else:
                    x, y = vowel_positions.get(chars[2], (-1, -1))
                    if x != -1:  # if the last letter is consonant
                        chars[2] = vowels[x][mark_idx]  # the mask is in the last letter
                    else:
                        # I think this fix Ubuntu unikey problems
                        # Or this is unnecessary since there is such a case like this
                        chars[1] = vowels[5][mark_idx] if chars[1] == 'i' else vowels[9][mark_idx]
                return ''.join(chars)
            return word

        for index in vowel_indexes:
            x, y = vowel_positions[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = vowels[x][mark_idx]
                # for index2 in nguyen_am_index:
                #     if index2 != index:
                #         x, y = nguyen_am_to_ids[chars[index]]
                #         chars[index2] = bang_nguyen_am[x][0]
                return ''.join(chars)

        # If there are more than two vowels in the word
        if len(vowel_indexes) == 2 and vowel_indexes[-1] == len(chars) - 1:
            # If the last character is vowel, the mark should be at the first vowel
            x, y = vowel_positions[chars[vowel_indexes[0]]]
            chars[vowel_indexes[0]] = vowels[x][mark_idx]
            # The mark should be at the second vowel
        else:
            # The mark should be at the second vowel
            x, y = vowel_positions[chars[vowel_indexes[1]]]
            chars[vowel_indexes[1]] = vowels[x][mark_idx]

        return ''.join(chars)

    @staticmethod
    def reformat(text):
        for punc in punctuation:
            text = text.replace(f"{punc}_", f"{punc} ")
            text = text.replace(f"_{punc}", f" {punc}")
        return text

    def normalize(self, text: str) -> str:
        text = self.convert_unicode(self.reformat(text))
        words = text.split()
        for i, word in enumerate(words):
            # Split punctuation at the beginning and after word
            parts = word.split('_')

            for j, part in enumerate(parts):
                components = self.component_pattern.sub(r'\1!@#\2!@#\3', part).split('!@#')
                if len(components) == 3:
                    mask = [x.isupper() for x in components[1]]
                    result = self.normalize_mark(components[1].lower())
                    new_word = ''
                    for m, r in zip(mask, result):
                        if m:
                            r = r.upper()
                        new_word += r
                    components[1] = new_word
                parts[j] = ' '.join(components).strip()
            concat = True
            for idx in range(1, len(parts)):
                if parts[idx][0].isupper() and parts[idx - 1].islower():
                    concat = False
                    break

            if concat:
                words[i] = '_'.join(parts)
            else:
                words[i] = ' '.join(parts)

        return ' '.join(words)


class TextNormalizer:
    vn_text_normalize = VietnameseTextNormalizer()
    non_escape_space = re.compile(r'\s+')
    valid_chars = re.compile(r"[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~aàáảãạâầấẩẫậăằắẳẵặeèéẻẽẹêềếểễệ"
                             r"iìíỉĩịoòóỏõọồốôổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵAÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆ"
                             r"IÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ0123456789"
                             r"bcdđfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZĐ\s]+")
    standard_mapping = {
        "…": "...",
        "–": "-",
        "“": "\"",
        "”": "\"",
    }
    punct = punctuation.replace(",", "").replace(".", "")
    punc_map = {p: f" {p} " for p in punct}
    punc_pattern = re.compile("[" + re.escape(punct) + "]")

    special_chars_pattern = re.compile(r"[…–“”]")

    def map_non_standard_character(self, text: str) -> str:
        """
        Replace some special characters with its standard character
        """
        text = self.non_escape_space.sub(" ", text)
        return self.special_chars_pattern.sub(lambda x: self.standard_mapping[x.group()], text)

    def remove_emoji(self, text: str) -> str:
        return ' '.join(self.valid_chars.findall(text))

    def normalize_punctuation(self, text: str) -> str:
        return self.punc_pattern.sub(lambda x: self.punc_map[x.group()], text)

    def normalize(self, text: str, norm_punct: bool = False, norm_mark: bool = False) -> str:
        """
        Normalize text

        Params:
            - text: input text
            - norm_punct: add spaces before and after punctuation
            - norm_mark: normalize vietnamese mark. E.g: Thuỵ -> Thụy
        Return:
            - normalized text
        """
        text = self.remove_emoji(text.strip())
        text = self.map_non_standard_character(text)

        if norm_mark:
            text = self.vn_text_normalize.normalize(text)

        if norm_punct:
            text = self.normalize_punctuation(text)
        return " ".join(text.split())

    @staticmethod
    def masking(text: str) -> Tuple[str, List[str]]:
        """
        Mask word that contains '_'
        """
        masked_words = []
        words = text.split()
        for i, word in enumerate(words):
            if '_' in word:
                masked_words.append(word)
                words[i] = 'MASK'

        return ' '.join(words), masked_words


class ModelUtils(metaclass=SingletonMeta):
    def __init__(
            self,
            input_max_length: int = None,
            tokenizer=None
    ):
        self.input_max_length = input_max_length
        ans_patterns = json.load(open(SPECIAL_TOKENS_PATH))[SPECIAL_TOKENS]
        ans_patterns = list(map(lambda x: x.upper(), ans_patterns))
        self.ans_patterns = re.compile("<(" + "|".join(ans_patterns) + ")>")

        special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[SPECIAL_TOKENS] + [e.name for e in QuestionType]
        self.special_tokens = list(map(lambda x: x.upper(), special_tokens))
        self.special_pattern = re.compile("<(" + "|".join(special_tokens) + ")>")
        self.tokenizer = tokenizer

        similar_entity_tag = json.load(open(SIMILAR_ENTITY_TAG, "r", encoding="utf8"))["similar"]
        self.similar_entity_tag = self.process_similar_tag(similar_entity_tag)

        self.normalizer = TextNormalizer()

    def process_similar_tag(self, tag_lst):
        return {
            tag: similar_tag_lst
            for similar_tag_lst in tag_lst
            for tag in similar_tag_lst
        }

    @staticmethod
    def generate_hash(data):
        try:
            json_string = json.dumps(data, sort_keys=True)
            hash_string = hashlib.md5(json_string.encode("utf-8")).hexdigest()
        except Exception as e:
            hash_string = ""
        return hash_string

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

        target_idx = None
        tokenized_passage = self.tokenizer.tokenize(" ".join(out_passage))
        out_passage = " ".join(tokenized_passage).split(" . ")
        while len(" . ".join(out_passage).split()) > self.input_max_length:
            retain_idx = [*range(len(out_passage))]
            if target_idx is None:
                for idx, ele in enumerate(out_passage):
                    if self.ans_patterns.search(ele):
                        target_idx = idx
                        break

            if not retain_idx or target_idx is None:
                logger.info(out_passage)
                logger.info(len([e for ele1 in out_passage for e in ele1.split()]))
                break
            retain_idx.remove(target_idx)
            # retain_idx.remove(target_idx)
            # target_idx = target_idx[-1]
            if not retain_idx:
                logger.info("retain_idx is empty")
                break
            rm_index = None
            if len(retain_idx) == 1:
                rm_index = retain_idx.pop(0)
            elif retain_idx[-1] < target_idx:
                rm_index = retain_idx.pop(0)
                target_idx -= 1
            elif retain_idx[0] > target_idx:
                rm_index = retain_idx.pop(-1)
            else:
                rm_index = retain_idx.pop(random.choice([0, -1]))
                if rm_index < target_idx:
                    target_idx -= 1
            if rm_index is None:
                logger.info("rm_index is NONE")
                break
            out_passage.pop(rm_index)

        return self.tokenizer.convert_tokens_to_string(" . ".join(out_passage).split()) + " ."

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
        passage_ans_clue = passage[:ans_lst[1]] + answer_chunk + passage[ans_lst[2]:]
        # split_passage = passage.split(" . ")
        # passage_ans_clue = " ".join( f"{ele} ." for ele in split_passage)
        p = f"<{ques_type}> {passage_ans_clue}" if ques_type else passage_ans_clue
        passage_ans_clue = self.truncate_passage(passage=p)
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
            if bar.n == bar.total:
                e.set()
                break

        # return passage_ans_clue

    def question_validation(self, data: BaseQGData, score: int = 0.4):
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
        passage = data.passage
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
        answer_pred = self.qa_api(context=passage, question=data.question)["data"][ANSWER].replace("_", " ").replace(
            "₫ ",
            "₫").replace(
            " %", "%").split()
        # except:
        #     return False
        # answer_truth = answer.replace("_", " ").split()
        # logger.info(answer_pred, answer_truth)
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
        if depth == 0:
            passage = self.normalizer.normalize(passage)
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

    def is_similar_tag(self, src_tag: str, dst_tag: str):
        return dst_tag in self.similar_entity_tag.get(src_tag, [src_tag])

    def concat_adjacent_entities(self, ner_dict: dict, passage: str):
        output_ner_dict = {}
        ner_list = [(ner_text, ner_lst) for ner_text, ner_lst in ner_dict.items()]
        ner_list.sort(key=lambda ele: ele[1][1])

        temp_lst = []
        for idx in range(len(ner_list)):
            start_pos = ner_list[idx][1][1]
            end_pos = ner_list[idx][1][2]
            tag = ner_list[idx][1][0]
            if temp_lst and self.is_similar_tag(src_tag=tag, dst_tag=temp_lst[0]) \
                    and (temp_lst[2] + 1 == start_pos or self._is_linkable_char(passage[temp_lst[2]:start_pos])):
                temp_lst = [tag, temp_lst[1], end_pos]
            elif not temp_lst:
                temp_lst = ner_list[idx][1]
            else:
                output_ner_dict[passage[temp_lst[1]: temp_lst[2]]] = temp_lst
                temp_lst = ner_list[idx][1]
        if temp_lst:
            output_ner_dict[passage[temp_lst[1]: temp_lst[2]]] = temp_lst
        return output_ner_dict

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
        assert not is_segmented_list or isinstance(passage, list), "ERROR!!!!!!!!"
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
            processed_passage = [passage]

        out_passage = ""
        count = 0
        for sub_passage in processed_passage:
            if not isinstance(sub_passage, str):
                logger.info(processed_passage)
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

        try:
            child_lst = sorted(node["children"], key=lambda t: t["id"])
        except:
            logger.info(node)
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

        return parsing_tree  # parsing_tree.get("ROOT", parsing_tree) if isinstance(parsing_tree, dict) else parsing_tree

    def get_chunk(self, passage, tag_lst: list = None, is_segmented: bool = False):
        """return chunks of sentence

        Args:
            passage (): __
            tag_lst (list)
            is_segmented (bool)

        Returns:
            list: list of chunks in sentence. [NER_tag, POS_tag, chunk, start_position, end_position]
        """
        if not tag_lst:
            tag_lst = ["AP", "PP-MNR"]
        if is_segmented:
            sentence_list = [" ".join(ele) for ele in passage]
        else:
            if isinstance(passage, list):
                passage = " ".join(passage).replace("_", " ")
            sentence_list = [" ".join(ele) for ele in Config.vncore_nlp.tokenize(passage)]

        index_sentence_in_passage = np.concatenate(
            [[0], np.cumsum([len(ele.split()) for ele in sentence_list])])
        _passage = " ".join(sentence_list).replace("_", " ")
        try:
            tree = self.parse_sentence(passage=_passage)
        except Exception as e:
            logger.error(f">> ERROR WHEN PARSE ALL PASSAGE << | {e}")
            tree = []
            for sub in sentence_list:
                try:
                    sub_ = self.parse_sentence(passage=sub.replace("_", " "))
                except:
                    sub_ = [{}]
                tree += sub_

        if len(tree) + 1 != len(index_sentence_in_passage):
            logger.error(
                f"len(tree)+1 should equal len(index_sentence_in_passage), but {len(tree)} + 1 != {len(index_sentence_in_passage)}")

        def process_sub_tree(sub_tree, start_idx):
            chunk_list = []
            _, orig_chunk_list = self._navigate(sub_tree, tag_lst=tag_lst)
            for ele in orig_chunk_list:
                try:
                    chunk = [re.sub(r"___\d+", "", e) for e in ele[:-1]]

                    start_pos = int(ele[0].split("___")[-1]) - 1
                    end_pos = int(ele[-2].split("___")[-1]) - 1
                    chunk_list.append((None, ele[-1], chunk, int(start_pos + start_idx),
                                       int(end_pos + start_idx)))
                except Exception as e:
                    logger.info(f'Exception in Chunking: {e}')
                    continue

            return chunk_list

        output = []
        for idx, sub_tree in enumerate(tree):
            if sub_tree == {}:
                continue
            output += process_sub_tree(sub_tree=sub_tree["ROOT"], start_idx=index_sentence_in_passage[idx])
        return output


if __name__ == "__main__":
    utils = ModelUtils(input_max_length=512, tokenizer=None)
    print(utils.get_chunk(
        "Một quả bóng quần vợt hay bóng tennis là một quả bóng có độ nẩy khi va đập , thiết kế cho môn thể thao quần vợt , nhưng cũng được dùng cho một số môn thể thao khác như squash tennis hay lotball . Những quả bóng quần vợt đầu tiên trong lịch sử được làm bằng da nhồi lông hay len . Từ thế kỷ 18 trở đi , một dải len rộng ¾ inch được quấn chặt quanh một nhân , rồi được buộc xung quanh bằng các sợi dây và bọc bên ngoài bằng vải trắng . Loại bóng này , được cải tiến với lõi gỗ bấc , vẫn được dùng trong môn real tennis ngày nay . Những năm 1870 , luật quần vợt quy định dùng cao su làm bóng , và các quả bóng được xếp trong ống khi vận chuyển , thường mỗi ống chứa bốn quả . Một quả bóng quần vợt hiện đại thông thường bao giờ cũng gồm hai phần , phần ruột và vỏ . Phần ruột được làm từ cao su rỗng ( lõi ) và phần vỏ phủ ra bên ngoài là chất liệu len ( nỉ ) . Hiện nay , bóng tennis có hai màu chính được phép sử dụng ở các giải đấu là trắng và vàng xanh . Bóng tennis có đường kính từ 2,5 inch ( 6,25 cm ) đến 2,63 inch ( 6,57 cm ) và có trọng lượng trong khoảng từ 56 gam đến 59,4 gam . Theo những quy định trong luật tennis , khi được thả từ độ cao 100 inch ( 254 cm ) xuống nền xi măng , bóng phải có độ nảy từ 53 đến 58 inch ( 135 đến 147 cm ) .",
        tag_lst=["QP"]))

