"""
Base Scheme class module
"""

import functools
import time
import logging

def timer(func):
    """
    Template docstring (EDIT)

    Args:
        arg1: arg1 decsription
    Returns:
        what function returns
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        logger = logging.getLogger()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        # line = f"{func.__qualname__} took {runtime//60} min {runtime%60:.4f} secs"
        logger.debug(
            "%s took %d min %.4f secs",
            func.__qualname__,
            runtime//60,
            runtime%60
        )
        return result

    return _wrapper
