"""The ``cache_utils`` module defines numpy compatible function wrappers
for implementing memoization.

Module API
----------
"""

import inspect
import sys
from collections import OrderedDict
from functools import wraps
from typing import *

import numpy as np


class MemoryCache(OrderedDict):
    """Ordered dictionary with an imposed limit on overall memory usage"""

    def __init__(self, max_size: int = None):
        """Ordered dictionary with an imposed size limit im memory

        When memory usage exceeds the predefined amount, remove the oldest
        entry from the cache.

        Args:
            max_size: Maximum memory size in bytes
        """

        super(MemoryCache, self).__init__()
        self.max_size = max_size
        if self.max_size and self.max_size <= (min_size := sys.getsizeof(self)):
            raise RuntimeError(f'Dictionary size limit must exceed {min_size} bytes')

    def __setitem__(self, key: Hashable, value: Any):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self) -> None:
        if self.max_size is not None:
            while sys.getsizeof(self) > self.max_size:
                self.popitem(last=False)


def numpy_cache(*numpy_args: str, cache_size: int = None):
    """Memoization decorator supporting ``numpy`` arrays

    Args:
        *numpy_args: Function arguments to treat as numpy arrays
        cache_size: Maximum memory to allocate to cached data in bytes

    Returns:
        A callable function decorator
    """

    def decorator(function: Callable):
        class Memoization(MemoryCache):
            """Dictionary like object that stores recent function calls in memory"""

            @wraps(function)
            def wrapped(self, *args: Any, **kwargs: Any) -> Any:
                """Wrapped version of the given function

                Arguments and returns are the same as ``function``
                """

                kwargs_for_key = inspect.getcallargs(function, *args, **kwargs)
                for arg_to_cast in numpy_args:
                    kwargs_for_key[arg_to_cast] = np.array(kwargs_for_key[arg_to_cast]).tostring()

                key = tuple(kwargs_for_key.items())
                try:
                    out = self[key]

                except KeyError:
                    out = self[key] = function(*args, **kwargs)

                return out

        return Memoization(max_size=cache_size).wrapped

    return decorator
