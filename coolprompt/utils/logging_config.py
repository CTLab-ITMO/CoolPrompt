from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import re


def setup_logging(logs_dir: str | Path = None) -> logging.Logger:
    """Logging config for CoolPrompt.

    Args:
        logs_dir: Optional log-saving directory. If None, only console
            logging is configured and no log file is created.
    """

    logger = logging.getLogger("coolprompt")
    resolved_logs_dir = Path(logs_dir).expanduser().resolve() if logs_dir else None
    if (
        getattr(logger, "_is_configured", False)
        and getattr(logger, "_logs_dir", None) == resolved_logs_dir
    ):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s.%(funcName)s] - %(message)s"
    )

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    if resolved_logs_dir is not None:
        os.makedirs(resolved_logs_dir, exist_ok=True)
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(resolved_logs_dir, f"run_{current_date}.log"),
            when="MIDNIGHT",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger._is_configured = True
    logger._logs_dir = resolved_logs_dir

    return logger


logger = setup_logging()


def set_verbose(verbose: int) -> None:
    """Sets the provided verbose level to the logger.

    Args:
        verbose (int): specifies the logging level
            0 - ERROR (only errors)
            1 - INFO (basic info + errors)
            2 - DEBUG (all messages)
    """

    logger_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[
        verbose
    ]
    logger.setLevel(logger_level)
    for handler in logger.handlers:
        handler.setLevel(logger_level)
