
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
