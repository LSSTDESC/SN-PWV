"""Tests for the ``caching`` module"""

import sys
from unittest import TestCase

import numpy as np

from snat_sim.utils.caching import Cache
from snat_sim.utils.caching import MemoryCache


class MemoryManagement(TestCase):
    """Tests for the enforcement of a limit on the consumed memory"""

    def test_max_size_is_enforced(self) -> None:
        """Test the memory cache enforces the prescribed size limit as data is added / removed"""

        # Create a cache that is just big enough to hold a single integer
        temp_cache = MemoryCache()
        temp_cache['a'] = 1
        test_cache = MemoryCache(max_size=sys.getsizeof(temp_cache))
        del temp_cache

        # Fill the cache to the limit and check the data has been saved
        test_cache['a'] = 1
        self.assertIn('a', test_cache, 'Data was not stored in the cache')

        # Exceed the memory limit and check the old data was deleted in favor of the new data
        test_cache['b'] = 1
        self.assertNotIn('a', test_cache, 'Old data was not deleted from the cache')
        self.assertIn('b', test_cache, 'New data was not stored in the cache')


class NumpyCache(TestCase):
    """Tests for the ``Cache`` function"""

    def test_number_function_evaluations(self) -> None:
        """Test the function is only evaluated once to store cache values"""

        call_count = 0

        @Cache()
        def foo():
            """A cached function that counts how many times it was run"""

            nonlocal call_count
            call_count += 1

        # Call the function twice and check it is only evaluated once
        foo()
        foo()
        self.assertEqual(call_count, 1)

    def test_supports_numpy_args(self) -> None:
        """Test the caching implimentation can handle numpy array arguments"""

        @Cache('x', 'y', cache_size=1000)
        def add(x: np.array, y: np.array) -> np.array:
            """Add two numpy arrays"""

            return x + y

        add(np.array([1, 2]), np.array([1, 2]))


class InitErrors(TestCase):
    """Tests for the raising of errors on init"""

    def test_error_on_zero_size_limit(self) -> None:
        """Test an error is raise when instantiating a MemoryCache of size zero"""

        with self.assertRaises(ValueError):
            MemoryCache(max_size=0)

    def test_error_on_negative_size_limit(self) -> None:
        """Test an error is raise when instantiating a MemoryCache of negative size"""

        with self.assertRaises(ValueError):
            MemoryCache(max_size=-1)

    def test_error_on_float_size_limit(self) -> None:
        """Test an error is raise when instantiating a MemoryCache with a float"""

        with self.assertRaises(ValueError):
            MemoryCache(max_size=1.2)
