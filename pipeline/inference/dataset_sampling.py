import argparse
import json
import random

import regex as re
import pandas as pd
from tqdm import tqdm
from threading import Thread, Event
from queue import Queue

from common.common_keys import *
from common.constants import *
from common.utils import *
from common.config import SingletonMeta, SamplingType, PipelineConfig
from pipeline.inference.sampling_pipeline import QuestionSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


def sampling_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_type', default="shopee", type=str,
                        help='dataset name to sampling')
    parser.add_argument('--return_entity', action='store_true',
                        help='whether or not return entity in sampling output')
    parser.add_argument('--verify', action='store_true',
                        help='whether or not verify output question')
    parser.add_argument('--need_sampling', action='store_true',
                        help='whether need sampling or not')
    parser.add_argument('--output_folder', default=SAMPLING_FOLDER + "/wiki_sampling/samplings_21_3", type=str)
    parser.add_argument('--model_device', default="cpu", type=str)

    parser.add_argument("--folder_checkpoint", default=OUTPUT_PATH + "/checkpoint/checkpoint_bart_19_3/", type=str)
    parser.add_argument('--input_max_length', default=512, type=int,
                        help='maximum context token number')
    parser.add_argument('--output_max_length', default=256, type=int,
                        help='maximum context token number')
    parser.add_argument('--parallel_input_processing', action='store_true')
    parser.add_argument('--inference_batch_size', default=4, type=int)
    parser.add_argument("--training_logging_dir", default=f"{OUTPUT_PATH}/logging/logging_bart_7_3/", type=str,
                        help="Tensorboard Logging Folder")

    return parser.parse_args()


class QASampling(metaclass=SingletonMeta):
    def __init__(self, args_config):
        self.config = args_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sampling_type = SamplingType(self.config.sampling_type)

        self.TYPE_TO_SAMPLING_FUNC = {
            SamplingType.SHOPEE: self.shopee_sampling,
            SamplingType.WIKI: self.wiki_sampling,
            SamplingType.TGDD: self.tgdd_sampling,
            SamplingType.SQUAD: self.SQuAD_sampling,
            SamplingType.TINHTE: self.tinhte_sampling,
        }
        self.output_folder = self.config.output_folder
        check_exist_folder(self.output_folder)

        self.sampler = QuestionSampler(PipelineConfig(training_output_dir=self.config.folder_checkpoint,
                                                      pipeline_input_max_length=self.config.input_max_length,
                                                      pipeline_output_max_length=self.config.output_max_length,
                                                      pipeline_device=self.config.model_device,
                                                      sampling_parallel_input_processing=self.config.parallel_input_processing,
                                                      sampling_inference_batch_size=self.config.inference_batch_size,
                                                      training_logging_dir=self.config.training_logging_dir))
        self.sampler_func = self.TYPE_TO_SAMPLING_FUNC[self.sampling_type]

        self.input_examples = self.get_raw_examples()

        self.input_queue = Queue(maxsize=10000)
        self.sampling_bar = tqdm(total=len(self.input_examples), initial=0, leave=True)

        self.sampling_event = Event()
        self.sampling_event.clear()

        self.threads = [Thread(target=self.qa_sampling, args=(), daemon=True) for _ in range(5)]

        [thread.start() for thread in self.threads]

    def get_raw_examples(self):
        self.input_folder = self.get_input_sampling_folder(sampling_type=self.sampling_type)
        all_files = os.listdir(self.input_folder)
        all_files = [f"{self.input_folder}/{f}" for f in all_files if f.endswith(".txt")]
        return all_files

    def qa_sampling(self):
        while not self.sampling_event.is_set() or not self.input_queue.empty():
            passage, output_file, _id = self.input_queue.get()
            samplings = self.sampler.sampling(passage=passage, _id=_id)
            if not samplings:
                self.logger.info(f"SAMPLING OUTPUT OF {_id} IS EMPTY!!!")

            json.dump(samplings, open(output_file.replace(self.input_folder, self.output_folder), "w", encoding="utf8"),
                      ensure_ascii=False,
                      indent=4)
            self.sampling_bar.update(1)

    def get_input_sampling_folder(self, sampling_type: SamplingType):
        input_folder = SAMPLING_FOLDER + f"/{sampling_type.value}_sampling/original/"
        if self.sampling_type.value == SamplingType.TGDD.value:
            input_folder = SAMPLING_FOLDER + f"/{sampling_type.value}_sampling/processed_data/"
        return input_folder

    def run(self):
        self.sampler_func()
        self.sampling_event.set()
        [thread.join() for thread in self.threads]

    def convert_samplings_to_SQuAD(self, output_file: str):
        """convert all sampling dataset to QA dataset

        Args:
            output_file (str): _description_
        """
        assert os.path.isdir(self.input_examples)
        all_files = os.listdir(self.input_examples)
        all_data = []
        for ele in tqdm(all_files):
            all_data += json.load(open(f"{self.input_examples}/{ele}", "r", encoding="utf8"))
        self.logger.info(all_data[-1])
        self.logger.info(f"Loaded {len(all_data)} examples")
        squad_dataset = []

        self.logger.info("Converting ... ")
        for idx, ele in enumerate(tqdm(all_data)):
            d = self.convert_QGexample_to_SQuADexample(ele, idx)
            if d:
                squad_dataset.append(d)
        self.logger.info(f"Converted {len(squad_dataset)} examples")
        json.dump({"data": squad_dataset}, open(output_file, "w", encoding="utf8"), ensure_ascii=False, indent=4)

    def wiki_sampling(self):
        done_data = []

        # with open(SAMPLING_FOLDER + "/wiki_sampling/wiki_examples.txt", "r", encoding="utf8") as f:
        #     done_data = f.readlines()
        # done_data = [e[:-1] for e in done_data]
        # self.logger.info("Done data:", len(done_data))

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

        for f in tqdm(self.input_examples):
            if f.replace(".txt", "") in done_data:
                continue

            with open(f, "r", encoding="utf8") as ff:
                data = ff.read().replace('\n', '.')
            passage = preprocess_passage(data)
            # if len(passage) > 10000:
            #     self.logger.info(3417414141343, len(passage))
            #     continue
            self.input_queue.put(
                (passage, f.replace(".txt", ".json"), f.replace(".txt", "").replace(self.input_folder, "")))

    def shopee_sampling(self):
        pass

    def tgdd_sampling(self):
        pass

    def tinhte_sampling(self):
        pass

    def SQuAD_sampling(self):
        pass


if __name__ == "__main__":
    config = sampling_config()
    sampling = QASampling(args_config=config)
    sampling.run()
