from typing import List, AnyStr
import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(logging_folder, log_name, log_exception_classes: List[AnyStr] = None, logging_level=logging.INFO):
    if log_exception_classes is None:
        log_exception_classes = []
    os.makedirs(logging_folder, exist_ok=True)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(name)s | [%(levelname)s] | %(lineno)d | %(message)s",
        handlers=[
            RotatingFileHandler(os.path.join(logging_folder, f"{log_name}.log"), encoding="utf8",
                                maxBytes=1024 * 10240, backupCount=20),
            logging.StreamHandler()
        ]
    )
    for exception_class in log_exception_classes:
        logging.getLogger(exception_class).setLevel(logging.WARNING)
