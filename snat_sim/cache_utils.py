"""The ``cache_utils`` module defines numpy compatible function wrappers
for implementing memoization.

Module API
----------
"""

from functools import lru_cache, wraps


def array_to_tuple(np_array):
    """Typecast a numpy array as a nested tuple"""

    try:
        return tuple(array_to_tuple(_) for _ in np_array)

    except TypeError:
        return np_array


# Credit: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array"""

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return cached_wrapper(*array_to_tuple(args), **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator
