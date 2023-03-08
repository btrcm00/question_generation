import os

SPECIAL_PATTERN = r"\.,:;\"!@%$'\]\[\(\)\{\}-"
URL_PATTERN = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+\|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
HASHTAG_PATTERN = r"#[^\s]+[\s+^\w\d]"
VIETNAMESE_RE = r"ÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"

CURRENT_PATH = os.getcwd()
OUTPUT_PATH = CURRENT_PATH + "/training_output/"
DATA_PATH = CURRENT_PATH + "/dataset/"
INFERENCE_PATH = CURRENT_PATH + "/pipeline/inference/"
EMB_CONFIG_PATH = DATA_PATH + "/emb_config.json"
TONE_MAPPING_PATH = DATA_PATH + "/tone_mapping.json"
SPECIAL_TOKENS_PATH = DATA_PATH + "/new_specials_tokens.json"
STOP_WORD_PATH = DATA_PATH + "stop_words.txt"
SAMPLING_FOLDER = DATA_PATH + "/sampling_dataset/"
TRAING_DATASET_FOLDER = DATA_PATH + "/training/"
stop_words = []  # open(STOP_WORD_PATH, "r", encoding="utf8").readlines()
STOP_WORDS_LIST = []  # [w[:-1] for w in stop_words]
