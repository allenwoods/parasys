import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        r = func(*args, **kwargs)
        end = time.process_time()
        print('Function: {}.{} \t Runtime: {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper