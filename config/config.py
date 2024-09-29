from pathlib import Path
import logging
import sys
import os

from dotenv import dotenv_values

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
REPORT_DIR = DATA_DIR / "reports"

ENV_VARIABLES = {
    **dotenv_values(BASE_DIR/".env"),  # load environment variables from .env file
    **os.environ,  # load environment variables from the system
}

def get_logger(name, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
