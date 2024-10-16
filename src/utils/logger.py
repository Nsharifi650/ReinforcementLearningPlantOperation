import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str | Path | None = None, level = logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger('ReinformentLearningForPlantOperation')

