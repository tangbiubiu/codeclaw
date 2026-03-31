import os
import sys

from dotenv import load_dotenv
from loguru import logger


def init_logger():
    load_dotenv()
    logger.remove()
    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    )

    if os.getenv("PROJECT_ENV") == "prod":
        logger.add(
            "logs/app_{time}.log",
            format=format_str,
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
        )

        logger.add(
            "logs/error_{time}.log",
            format=format_str,
            level="ERROR",
            rotation="100 MB",
            retention="30 days",
            enqueue=True,
            filter=lambda record: record["level"].no >= 40,
        )
    else:
        logger.add(
            sys.stderr,
            format=format_str,
            level="DEBUG",
        )

    return logger


app_logger = init_logger()
