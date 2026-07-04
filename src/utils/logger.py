# FIXED — configure once, reuse the same logger
import logging
import os
from datetime import datetime

_logger = None  # module-level singleton

def get_logger():
    global _logger
    if _logger is not None:
        return _logger

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("EduGenie")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    _logger = logger
    return _logger