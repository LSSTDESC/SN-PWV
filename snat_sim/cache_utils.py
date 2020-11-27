"""The ``cache_utils`` module defines numpy compatible function wrappers
for implementing memoization.

Module API
----------
"""

import sys
from collections import OrderedDict
from functools import wraps


class MemoryCache(OrderedDict):

    def __init__(self, max_size=None):
        """Ordered dictionary with an imposed size limit im memory

        When memory usage exceeds the predefined amount, remove the oldest
        entry from the cache.

        Args:
            max_size (int): Maximum memory size in bytes
        """

        super(MemoryCache, self).__init__()
        self.max_size = max_size
        if self.max_size and self.max_size <= (min_size := sys.getsizeof(self)):
            raise RuntimeError(f'Dictionary size limit must exceed {min_size} bytes')

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.max_size is not None:
            while sys.getsizeof(self) > self.max_size:
                self.popitem(last=False)


def fast_cache(*args, cache_size=None):
    """Memoization decorator supporting ``numpy`` arrays

    Wrapped functions do not accept positional arguments and must use keyword
    arguments only.

    Args:
        *args      (str): Function arguments to treat as numpy arrays
        cache_size (int): Maximum memory to allocate to cached data in bytes

    Returns:
        A callable function decorator
    """

    def decorator(function):
        class Memoization(MemoryCache):
            @wraps(function)
            def wrapper(self, **kwargs):
                kwargs_for_key = kwargs.copy()
                kwargs_for_key.update({a: kwargs_for_key[a].tostring() for a in args})
                key = tuple(kwargs_for_key.items())
                try:
                    out = self[key]

                except KeyError:
                    out = self[key] = function(**kwargs)

                return out

        return Memoization(max_size=cache_size).wrapper

    return decorator
