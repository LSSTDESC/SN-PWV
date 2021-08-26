"""The ``caching`` module defines numpy compatible function wrappers
for implementing function level memoization.

Usage Example
-------------

The builtin Python memoization routines (e.g., ``lru_cache``) are not
compatible with ``numpy`` arrays because array objects are not hashable.
The ``Cache`` decorator provides an alternative memoization solution
that supports numpy arguments. Arguments that are numpy arguments must be
specified by name when constructing the decorator:

.. doctest:: python

   >>> import numpy as np

   >>> from snat_sim.utils.caching import Cache

   >>> def add(x: np.array, y: np.array) -> np.array:
   ...     print('The function has been called!')
   ...     return x + y

   >>> add = Cache(add, 1000, 'x', 'y')
   >>> x_arr = np.arange(1, 5)
   >>> y_arr = np.arange(5, 9)

   >>> print(add(x_arr, y_arr))
   The function has been called!
   [ 6  8 10 12]

   >>> print(add(x_arr, y_arr))
   [ 6  8 10 12]

Module API
----------
"""

from __future__ import annotations

import inspect
import sys
from collections import OrderedDict
from typing import Any, Callable, Hashable, Tuple, Type

import numpy as np


class MemoryCache(OrderedDict):
    """Ordered dictionary with an imposed limit on overall memory usage"""

    def __init__(self, max_size: int = None):
        """Ordered dictionary with an imposed size limit im memory.

        When memory usage exceeds the predefined amount, remove the oldest
        entry from the cache.

        Args:
            max_size: Maximum memory size in bytes
        """

        super(MemoryCache, self).__init__()

        self.max_size = max_size
        if max_size is not None:
            if not isinstance(max_size, int) or max_size <= 0:
                raise ValueError('Maximum cache size must be a positive integer')

    def __setitem__(self, key: Hashable, value: Any):
        """Update an entry in the hash table."""

        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self) -> None:
        """Pop items from memory until instance size is <= the size limit."""

        if self.max_size is not None:
            while self and sys.getsizeof(self) > self.max_size:
                self.popitem(last=False)


class Cache(MemoryCache):
    """Memoization function wrapper"""

    def __init__(self, function: callable, cache_size: int, *numpy_args: str) -> None:
        """Memoization decorator supporting ``numpy`` arrays.

        Args:
            *numpy_args: Function arguments to treat as numpy arrays
            cache_size: Maximum memory to allocate to cached data in bytes

        Returns:
            A callable function decorator
        """

        self.function = function
        self.cache_size = cache_size
        self.numpy_args = numpy_args
        super(Cache, self).__init__(max_size=cache_size)

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """Cache return values of the given function

        Args:
            function: The function to cache returns of

        Returns:
            The wrapped function
        """

        kwargs_for_key = inspect.getcallargs(self.function, *args, **kwargs)
        for arg_to_cast in self.numpy_args:
            kwargs_for_key[arg_to_cast] = np.array(kwargs_for_key[arg_to_cast]).tobytes()

        key = tuple(kwargs_for_key.items())
        try:
            return self[key]

        except KeyError:
            new_val = self.function(*args, **kwargs)
            self[key] = new_val
            return new_val

    def __reduce__(self) -> Tuple[Type[Cache], Tuple]:
        # Ensures instances can be pickled
        return self.__class__, (self.function, self.cache_size, *self.numpy_args)
