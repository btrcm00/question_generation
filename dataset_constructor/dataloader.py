import torch
import random
import json
import os

import regex as re
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from common.common_keys import *
from common.utils import load_file
from common.config import QuestionType, PipelineConfig
from common.constants import SPECIAL_TOKENS_PATH


def get_ner_in_tensor(input_ids, ner_list, tokenizer):
    indices = torch.ones(len(input_ids))
    for ent in ner_list:
        ner = tokenizer(ent).input_ids[:-1][1:]
        list_indices_of_ner = [i for i in range(len(input_ids)) if input_ids[i:i + len(ner)] == ner]

        for idx in list_indices_of_ner:
            indices[idx:idx + len(ner)] = 1.5
    return indices


class FQG_dataset(Dataset):
    def __init__(self, config: PipelineConfig, mode='train', tokenizer=None, added_new_special_tokens=False,
                 model_type="marian"):
        super().__init__()
        assert mode in ["train", "dev", "test"], "train should be 'train', 'dev' or 'test'"

        self.mode = mode
        self.config = config

        if not tokenizer:
            assert config.pipeline_pretrained_path is not None, "You have to specify tokenizer path if tokenizer is None"
            self.tokenizer = AutoTokenizer.from_pretrained(config.pipeline_pretrained_path, use_fast=False)
        else:
            self.tokenizer = tokenizer

        if not added_new_special_tokens:
            new_special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[
                                     SPECIAL_TOKENS] + [e.name for e in QuestionType]
            new_special_tokens = list(set(new_special_tokens))
            special_tokens_dict = {"additional_special_tokens": [f"<{tk.upper()}>" for tk in new_special_tokens]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.data = self.load_data(path=f"{config.pipeline_dataset_folder}/processed", mode=mode)
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_item = self.data[idx]

        in_text = data_item[MODEL_INPUT].replace("\u200b", "")
        in_text = re.sub(r"( \.)+", " .", in_text)
        # ques_type_token = "<" + data_item[MODEL_QUESTION_TYPE_INPUT].upper().replace(" ", "_") + ">"
        # inputs = ques_type_token + " " + in_text

        passage_tokenized = self.tokenizer(in_text, padding="max_length",
                                           max_length=self.config.pipeline_input_max_length, truncation=True)

        processed_labels = data_item[MODEL_LABEL].replace("\u200b", "")
        labels = self.tokenizer(processed_labels, padding="max_length",
                                max_length=self.config.pipeline_output_max_length, truncation=True).input_ids  # [1:]
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]

        d = data_item[MODEL_ENTITY_DICT_INPUT].keys()
        entity_weight = get_ner_in_tensor(labels,
                                          ["_".join(e.split()) if "_".join(e.split()) in processed_labels else e for e
                                           in d], self.tokenizer)

        return {
            INPUT_IDS: passage_tokenized.input_ids,
            ATTENTION_MASK: passage_tokenized.attention_mask,
            ENTITY_WEIGHT: entity_weight,
            LABEL: labels
        }

    @staticmethod
    def load_data(path, mode):
        data = []
        for file in os.listdir(path):
            if mode in file:  # and file.endswith(".pkl"):
                data += load_file(path + file)
                data = [ele for ele in data if ele[MODEL_QUESTION_TYPE_INPUT].upper() not in ["OTHER", "BOOLEAN"]]
        random.shuffle(data)
        print(f"Loaded {len(data)} examples in {mode} dataset ...")
        if mode == "dev" or mode == "train":
            return data[:20]
        return data

    @staticmethod
    def get_dataset(config: PipelineConfig = None, mode=None, tokenizer=None, added_new_special_tokens=False,
                    model_type="marian"):
        tok = None
        if tokenizer:
            tok = tokenizer

        if mode:
            return FQG_dataset(config=config, mode=mode, tokenizer=tok)
        else:
            train_dataset = FQG_dataset(config=config, mode="train", tokenizer=tok,
                                        added_new_special_tokens=added_new_special_tokens, model_type=model_type)
            dev_dataset = FQG_dataset(config=config, mode="dev", tokenizer=tok,
                                      added_new_special_tokens=added_new_special_tokens, model_type=model_type)
            test_dataset = FQG_dataset(config=config, mode="test", tokenizer=tok,
                                       added_new_special_tokens=added_new_special_tokens, model_type=model_type)

            return train_dataset, dev_dataset, test_dataset
