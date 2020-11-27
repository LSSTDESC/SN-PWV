"""The ``cache_utils`` module defines numpy compatible function wrappers
for implementing memoization.

Module API
----------
"""

from functools import wraps


def fast_cache(*args):
    """Memoization decorator supporting ``numpy`` arrays

    Args:
        *args (str): Function arguments to treat as numpy arrays

    Returns:
        A callable function decorator
    """

    def decorator(function):
        class Memoization(dict):
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

        return Memoization().wrapper

    return decorator
