"""
Base Scheme class module
"""

import functools
import time


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
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        line = f"{func.__qualname__} took {runtime//60} min {runtime%60:.4f} secs"
        print(f"{time.strftime('%H:%M:%S')} - " + line)
        return result

    return _wrapper
