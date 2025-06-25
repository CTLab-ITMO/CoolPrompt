from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import re


def setup_logging():
    """Logging config for CoolPrompt"""

    logs_dir = Path(__file__).parents[2] / "logs"
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger("coolprompt")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s.%(funcName)s] - %(message)s"
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(logs_dir, f"run_{current_date}.log"),
        when="MIDNIGHT",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()
