"""Tests for the ``snat_sim.utils.caching.Cache`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.utils.caching import Cache


def test_func_to_be_wrapped(x: np.array, y: np.array) -> np.array:
    """Add two numpy arrays"""

    return x + y


class ResultsAreCached(TestCase):
    """Test the function is only evaluated once to store cache values"""

    def runTest(self) -> None:
        call_count = 0

        def foo() -> None:
            """A cached function that counts how many times it was run"""

            nonlocal call_count
            call_count += 1

        foo = Cache(foo, 1000)
        # Call the function twice and check it is only evaluated once
        foo()
        foo()
        self.assertEqual(call_count, 1)


class NumpySupport(TestCase):
    """Test the caching implementation can handle numpy array arguments"""

    @staticmethod
    def runTest() -> None:
        add = Cache(test_func_to_be_wrapped, 1000, 'x', 'y')
        add(np.array([1, 2]), np.array([1, 2]))


class ReductionMatchesInstanceArgs(TestCase):
    """Test the ``__reduce__`` method returns the correct identifier information"""

    def test_correct_class_returned(self) -> None:
        """Test the returned class matches the instance class"""

        returned_class, _ = Cache(test_func_to_be_wrapped, 1000, 'x', 'y').__reduce__()
        self.assertEqual(returned_class, Cache)

    def test_correct_args(self) -> None:
        """Test the returned args match the instantiation arguments"""

        args = test_func_to_be_wrapped, 1000, 'x', 'y'
        _, returned_args = Cache(*args).__reduce__()
        self.assertSequenceEqual(args, returned_args)
