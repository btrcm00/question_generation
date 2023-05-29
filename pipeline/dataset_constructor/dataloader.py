import copy

import torch
import random
import json
import os
import logging

import regex as re
from torch.utils.data import Dataset

from common.common_keys import *
from common.utils import load_file
from common.config import *
from common.constants import *


def get_ner_in_tensor(input_ids, ner_list, tokenizer):
    indices = torch.ones(len(input_ids))
    for ent in ner_list:
        ner = tokenizer(ent).input_ids[:-1][1:]
        list_indices_of_ner = [i for i in range(len(input_ids)) if input_ids[i:i + len(ner)] == ner]

        for idx in list_indices_of_ner:
            indices[idx:idx + len(ner)] = 1.5
    return indices


class FQG_dataset(Dataset):
    def __init__(self, tokenizer, config: PipelineConfig = None, mode='train'):
        super().__init__()
        assert mode in ["train", "dev", "test"], "train should be 'train', 'dev' or 'test'"

        self.mode = mode
        self.config = config if config is not None else PipelineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tokenizer = tokenizer
        self.data = self.load_data(path=f"{self.config.pipeline_dataset_folder}/processed")

        remove_tokens: list = json.load(open(REMOVE_TOKENS_PATH, "r", encoding="utf8"))["tokens"]
        self.remove_tokens_pattern = "|".join(remove_tokens)

    def __len__(self):
        return len(self.data)

    def pre_process(self, data: dict):
        clean_data = {}
        for key, value in data.items():
            clean_data[key] = re.sub(self.remove_tokens_pattern, "", value) if isinstance(value, str) else value
        return clean_data

    def __getitem__(self, idx: int):
        data_item = self.data[idx]
        data_item = self.pre_process(data_item)

        in_text = data_item[MODEL_INPUT]
        in_text = re.sub(r"( \.)+", " .", in_text)
        if data_item[MODEL_QUESTION_TYPE_INPUT].upper() == "OTHER" or random.randint(0, 1000) < 200:
            in_text = in_text.replace("<" + data_item[MODEL_QUESTION_TYPE_INPUT].upper() + "> ", "")

        passage_tokenized = self.tokenizer(in_text, padding="max_length",
                                           max_length=self.config.pipeline_input_max_length, truncation=True)

        processed_labels = data_item[MODEL_LABEL]
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

    def load_data(self, path: str):
        data = []
        for file in os.listdir(path):
            if self.mode in file and file.endswith(".pkl"):
                data += load_file(f"{path}/{file}")
                data = [ele for ele in data if ele[MODEL_QUESTION_TYPE_INPUT].upper() not in ["BOOLEAN", "OTHER"] or (
                        ele[MODEL_QUESTION_TYPE_INPUT].upper() == "OTHER" and random.randint(0, 1000) < 100)]
        random.shuffle(data)
        self.logger.info(f"Loaded {len(data)} examples in {self.mode} dataset ...")
        if self.mode in ["dev","test"]:
            return data[:2000]
        return data

    @staticmethod
    def get_dataset(tokenizer, config: PipelineConfig = None, mode: str = None):
        if mode is not None:
            return FQG_dataset(config=config, mode=mode, tokenizer=tokenizer)
        else:
            train_dataset = FQG_dataset(config=config, mode="train", tokenizer=tokenizer)
            dev_dataset = FQG_dataset(config=config, mode="dev", tokenizer=tokenizer)
            test_dataset = FQG_dataset(config=config, mode="test", tokenizer=tokenizer)

            return train_dataset, dev_dataset, test_dataset
