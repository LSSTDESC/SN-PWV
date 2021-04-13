"""Tests for the ``snat_sim.utils.caching.Cache`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.utils.caching import Cache


class NumpyResultsAreCached(TestCase):
    """Test the function is only evaluated once to store cache values"""

    def runTest(self) -> None:
        call_count = 0

        def foo():
            """A cached function that counts how many times it was run"""

            nonlocal call_count
            call_count += 1

        foo = Cache(foo, 1000)
        # Call the function twice and check it is only evaluated once
        foo()
        foo()
        self.assertEqual(call_count, 1)


class NumpySupport(TestCase):

    def runTest(self) -> None:
        """Test the caching implementation can handle numpy array arguments"""

        def add(x: np.array, y: np.array) -> np.array:
            """Add two numpy arrays"""

            return x + y

        add = Cache(add, 1000, 'x', 'y')
        add(np.array([1, 2]), np.array([1, 2]))
