import time
import functools
from src.utils.logger import get_logger

logger = get_logger()

def measure_time(func):
    @functools.wraps(func) #preserves __name__, __doc__
    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        logger.info(
            f"{func.__name__} executed in {end-start:.2f} sec"
        )

        return result

    return wrapper