"""
Logging setup.

Provides a single get_logger() function so every module logs in a consistent
format — to BOTH the console (live progress) and a file in logs/ (permanent,
timestamped record of every run).
"""

import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


def get_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Args:
        name: Name of the logger — usually the module's __name__.
        log_file: Path to the log file.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # If this logger already has handlers, it was configured before —
    # return it as-is to avoid attaching duplicate handlers
    # (which would print every log line multiple times).
    if logger.handlers:
        return logger

    # Defines HOW each log line looks: time | module | level | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler 1 — send logs to the console (standard output).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2 — also write logs to a file.
    # Ensure the logs/ folder exists before writing to it.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Quick self-test: python -m src.logger
if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("This is an INFO message ✅")
    log.warning("This is a WARNING message ⚠️")
    log.error("This is an ERROR message ❌")
