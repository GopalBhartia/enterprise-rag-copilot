import logging
import sys


def setup_logging():
    logger = logging.getLogger("rag")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# GLOBAL LOGGER (IMPORTANT)
logger = setup_logging()
