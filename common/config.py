import json
import random
from enum import Enum

from dotenv import load_dotenv
from vncorenlp import VnCoreNLP

from common.constants import *
from common.common_keys import *

load_dotenv()

random.seed(2023)


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    special_pattern = SPECIAL_PATTERN
    url_pattern = URL_PATTERN
    email_pattern = EMAIL_PATTERN
    hashtag_pattern = HASHTAG_PATTERN

    # Change config at .env
    service_host = os.getenv(SERVICE_HOST)
    vncore_url = os.getenv(VNCORE_URL)
    vncore_port = int(os.getenv(VNCORE_PORT))
    vncore_nlp = VnCoreNLP(address=vncore_url, port=vncore_port)

    parsing_url = os.getenv(PARSING_URL)
    ner_url = os.getenv(NER_URL)
    qa_url = os.getenv(QA_URL)
    augment_url = os.getenv(AUGMENT_URL)

    stop_word_lst = open(STOP_WORD_PATH, "r", encoding="utf8").readlines()
    stop_word_lst = [w[:-1] for w in stop_word_lst]

    ques_type_config = json.load(open(DATA_PATH + "/sampling_questype_ner.json", "r", encoding="utf8"))
    tone_mapping = json.load(open(DATA_PATH + "/tone_mapping.json", "r", encoding="utf8"))


class PipelineConfig:
    def __init__(self,
                 training_output_dir: str = None,
                 training_logging_dir: str = None,
                 training_batch_size: int = None,
                 training_weight_decay: float = None,
                 training_save_total_limit: int = None,
                 training_learning_rate: float = None,
                 training_gradient_accumulation_steps: int = None,
                 training_eval_steps: int = None,
                 training_logging_steps: int = None,
                 training_save_steps: int = None,
                 training_num_epochs: int = None,
                 training_restore_checkpoint: bool = None,
                 training_restore_folder: str = None,
                 training_generation_num_beams: int = None,
                 training_metrics: str = None,
                 training_warm_up_ratio: float = None,
                 training_use_pointer: bool = None,

                 pipeline_output_max_length: int = None,
                 pipeline_input_max_length: int = None,
                 pipeline_dataset_folder: str = None,
                 pipeline_pretrained_path: str = None,
                 pipeline_special_tokens_path: str = None,
                 pipeline_device: str = None,

                 constructor_num_of_threads: int = None,
                 constructor_ratio: list = None,

                 sampling_parallel_input_processing: bool = None,
                 sampling_inference_batch_size: int = None,
                 sampling_dataset_folder: str = None,
                 sampling_type: str = None,
                 sampling_verify: bool = None,
                 sampling_return_entity: bool = None
                 ):
        self.training_output_dir = training_output_dir if training_output_dir is not None \
                                       else os.getenv(TRAINING_OUTPUT_DIR, None)
        self.training_logging_dir = training_logging_dir if training_logging_dir is not None \
                                        else os.getenv(TRAINING_LOGGING_DIR, None)
        self.training_batch_size = training_batch_size if training_batch_size is not None \
                                       else int(os.getenv(TRAINING_BATCH_SIZE, "4"))
        self.training_weight_decay = training_weight_decay if training_weight_decay is not None \
                                         else float(os.getenv(TRAINING_WEIGHT_DECAY, "0.1"))
        self.training_save_total_limit = training_save_total_limit if training_save_total_limit is not None \
                                             else int(os.getenv(TRAINING_SAVE_TOTAL_LIMIT, "5"))
        self.training_learning_rate = training_learning_rate if training_learning_rate is not None \
                                          else float(os.getenv(TRAINING_LEARNING_RATE, "0.1"))
        self.training_gradient_accumulation_steps = training_gradient_accumulation_steps if training_gradient_accumulation_steps is not None \
                                                        else int(os.getenv(TRAINING_GRADIENT_ACCUMULATION_STEPS, "2"))
        self.training_eval_steps = training_eval_steps if training_eval_steps is not None \
                                       else int(os.getenv(TRAINING_EVAL_STEPS, "2000"))
        self.training_logging_steps = training_logging_steps if training_logging_steps is not None \
                                          else int(os.getenv(TRAINING_LOGGING_STEPS, "2000"))
        self.training_save_steps = training_save_steps if training_save_steps is not None \
                                       else int(os.getenv(TRAINING_SAVE_STEPS, "2000"))
        self.training_num_epochs = training_num_epochs if training_num_epochs is not None \
                                       else int(os.getenv(TRAINING_NUM_EPOCHS, "20"))
        self.training_restore_checkpoint = training_restore_checkpoint if training_restore_checkpoint is not None \
                                               else bool(os.getenv(TRAINING_RESTORE_CHECKPOINT, "1"))
        self.training_restore_folder = training_restore_folder if training_restore_folder is not None \
                                           else os.getenv(TRAINING_RESTORE_FOLDER, None)
        self.training_generation_num_beams = training_generation_num_beams if training_generation_num_beams is not None \
                                                 else int(os.getenv(TRAINING_GENERATION_NUM_BEAMS, "5"))
        self.training_metrics = training_metrics if training_metrics is not None \
                                    else os.getenv(TRAINING_METRICS, "bleu")
        self.training_warm_up_ratio = training_warm_up_ratio if training_warm_up_ratio is not None \
                                          else float(os.getenv(TRAINING_WARM_UP_RATIO, "0.1"))
        self.training_use_pointer = training_use_pointer if training_use_pointer is not None \
                                        else bool(os.getenv(TRAINING_USE_POINTER, "True"))

        self.pipeline_output_max_length = pipeline_output_max_length if pipeline_output_max_length is not None \
                                              else int(os.getenv(PIPELINE_OUTPUT_MAX_LENGTH, "256"))
        self.pipeline_input_max_length = pipeline_input_max_length if pipeline_input_max_length is not None \
                                             else int(os.getenv(PIPELINE_INPUT_MAX_LENGTH, "512"))
        self.pipeline_dataset_folder = pipeline_dataset_folder if pipeline_dataset_folder is not None \
                                           else os.getenv(PIPELINE_DATASET_FOLDER, None)
        self.pipeline_pretrained_path = pipeline_pretrained_path if pipeline_pretrained_path is not None \
                                            else os.getenv(PIPELINE_PRETRAINED_PATH)
        self.pipeline_special_tokens_path = pipeline_special_tokens_path if pipeline_special_tokens_path is not None \
                                                else os.getenv(PIPELINE_SPECIAL_TOKENS_PATH, None)
        self.pipeline_device = pipeline_device if pipeline_device is not None \
                                   else os.getenv(PIPELINE_DEVICE, "cpu")

        self.constructor_num_of_threads = constructor_num_of_threads if constructor_num_of_threads is not None \
                                              else int(os.getenv(CONSTRUCTOR_NUM_OF_THREADS, "5"))
        self.constructor_ratio = constructor_ratio if constructor_ratio is not None \
                                     else os.getenv(CONSTRUCTOR_RATIO)
        count = 0
        while self.constructor_ratio is not None and not isinstance(self.constructor_ratio, list) and count < 5:
            self.constructor_ratio = eval(self.constructor_ratio)
            count += 1

        self.sampling_parallel_input_processing = sampling_parallel_input_processing if sampling_parallel_input_processing is not None \
                                                      else bool(os.getenv(SAMPLING_PARALLEL_INPUT_PROCESSING, "1"))
        self.sampling_inference_batch_size = sampling_inference_batch_size if sampling_inference_batch_size is not None \
                                                 else int(os.getenv(SAMPLING_INFERENCE_BATCH_SIZE, "4"))
        self.sampling_dataset_folder = sampling_dataset_folder if sampling_dataset_folder is not None \
                                           else os.getenv(SAMPLING_DATASET_FOLDER, None)
        self.sampling_type = sampling_type if sampling_type is not None \
                                 else os.getenv(SAMPLING_TYPE, None)
        self.sampling_verify = sampling_verify if sampling_verify is not None \
                                   else bool(os.getenv(SAMPLING_VERIFY, ""))
        self.sampling_return_entity = sampling_return_entity if sampling_return_entity is not None \
                                          else bool(os.getenv(SAMPLING_RETURN_ENTITY, "1"))


class QuestionType(Enum):
    WHO = 1
    WHERE = 2
    WHEN = 3
    WHY = 4
    WHICH = 5
    WHAT = 6
    HOW_MANY = 7
    HOW_FAR = 8
    HOW_LONG = 9
    HOW = 10
    OTHER = 11


class ModelInputTag:
    clue = "<CLUE>"
    close_clue = "</CLUE>"
    answer = "<ANS>"
    close_answer = "</ANS>"


class SamplingType(Enum):
    SHOPEE = "shopee"
    WIKI = "wiki"
    TGDD = "tgdd"
    SQUAD = "squad"
    TINHTE = "tinhte"
