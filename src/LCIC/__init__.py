import os
import sys
import logging
import time

DATASET_1: str = "https://www.kaggle.com/datasets/gpiosenka/sports-classification"
DATASET_2: str = "https://www.kaggle.com/datasets/gpiosenka/headgear-image-classification"
DATASET_3: str = "https://www.kaggle.com/datasets/gpiosenka/100-bird-species"
DATASET_4: str = "https://www.kaggle.com/datasets/pavansanagapati/images-dataset"
DATASET_5: str = "https://www.kaggle.com/datasets/gpiosenka/balls-image-classification"

logging_str = "[%(asctime)s]: %(levelname)s: %(module)s: %(message)s"
log_dir = "logs"
log_filesPath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_str,
                    handlers=[
                        logging.FileHandler(log_filesPath),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger("LCIC")


def get_unique_file_name(prefix: str, ext: str = "log") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
