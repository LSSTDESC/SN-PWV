"""Tests for the ``snat_sim.utils.caching.MemoryManagement`` class"""

import sys
from unittest import TestCase

from snat_sim.utils.caching import MemoryCache


class MemoryManagement(TestCase):
    """Tests for the enforcement of a limit on the consumed memory"""

    def runTest(self) -> None:
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
            # noinspection PyTypeChecker
            MemoryCache(max_size=1.2)
