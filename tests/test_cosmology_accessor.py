"""Tests for the ``CosmologyAccessor`` class"""

from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd

from snat_sim import constants as const
from snat_sim.fitting_pipeline import CosmologyAccessor
from tests import mock


class AccessorRegistration(TestCase):
    """Test the ``snat_sim`` accessor is registered with ``pandas``"""

    def runTest(self):
        self.assertTrue(pd.DataFrame().snat_sim)


class DistanceModulus(TestCase):
    """Tests for the chi-squared minimization"""

    def setUp(self):
        self.test_data = mock.create_mock_pipeline_outputs()

    def test_returns_distance_modulus(self):
        """Test the returned value matches the distance modulus"""

        abs_mag = -19.1
        dist_mod = self.test_data['mb'] - abs_mag
        returned = self.test_data.snat_sim.calc_distmod(abs_mag)
        pd.testing.assert_series_equal(returned, dist_mod)


class MCResampling(TestCase):
    """Tests for the chi-squared minimization"""

    def setUp(self):
        self.test_data = mock.create_mock_pipeline_outputs()

        # Arguments to ensure the minimization converges and quickly
        self.iminuit_kwargs = dict(
            H0=const.betoule_H0,
            Om0=const.betoule_omega_m,
            limit_Om0=(0, 1),
            w0=-1,
            fix_w0=True,
            abs_mag=const.betoule_abs_mb,
            fix_abs_mag=True
        )

    def test_correct_number_of_samples(self):
        """Test the correct number of samples are drawn from the dataset"""

        num_samples = 3
        samples = self.test_data.snat_sim.minimize_mc(num_samples, frac=1, **self.iminuit_kwargs)
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), num_samples)

    def test_statistic_returns_dict(self):
        """Test supplying a statistic argument returns a dict of the applies statistic"""

        num_samples = 3
        stat_func = partial(np.average, axis=0)
        samples = self.test_data.snat_sim.minimize_mc(num_samples, frac=1, **self.iminuit_kwargs)
        statistic = self.test_data.snat_sim.minimize_mc(num_samples, frac=1, statistic=stat_func, **self.iminuit_kwargs)

        self.assertIsInstance(statistic, dict)

        # Test the dictionary is built  with correct key value mapping
        for param in ('H0', 'Om0', 'abs_mag', 'w0'):
            np.testing.assert_allclose(statistic[param], stat_func([s.values[param] for s in samples]))


class ChisqGrid(TestCase):
    """Tests for the ``chisq_grid`` function"""

    def setUp(self):
        self.test_data = mock.create_mock_pipeline_outputs()

    def test_grid_values_match_chisq_function(self):
        """Test returned grid values match the ``chisq`` calculation"""

        # Calculate chisq for two different w0 values so chisq values are also different
        chisq1 = self.test_data.snat_sim.chisq(
            w0=-1,
            H0=const.betoule_H0,
            Om0=const.betoule_omega_m,
            abs_mag=const.betoule_abs_mb,
            alpha=0,
            beta=0
        )

        chisq2 = self.test_data.snat_sim.chisq(
            w0=0,
            H0=const.betoule_H0,
            Om0=const.betoule_omega_m,
            abs_mag=const.betoule_abs_mb,
            alpha=0,
            beta=0
        )

        chisq_grid = self.test_data.snat_sim.chisq_grid(
            w0=[-1, 0],
            H0=const.betoule_H0,
            Om0=const.betoule_omega_m,
            abs_mag=const.betoule_abs_mb,
            alpha=0,
            beta=0)

        np.testing.assert_allclose(chisq_grid, [chisq1, chisq2])


class MatchArgumentDimensions(TestCase):
    """Tests for the ``_match_argument_dimensions`` function"""

    @staticmethod
    def assert_return(inputs, expected):
        """For input arguments ``input`` assert the returned array matches ``expected``

        Args:
            inputs  (tuple): Arguments for ``_match_argument_dimensions``
            expected (list): Expected return values
        """

        returned = CosmologyAccessor._match_argument_dimensions(*inputs)
        np.testing.assert_array_equal(returned, expected)

    def assert_return_matching_dimensions(self, inputs):
        """For input arguments ``input`` assert the returned array matches ``list(input)``

        Args:
            inputs  (tuple): Arguments for ``_match_argument_dimensions``
        """

        self.assert_return(inputs, list(inputs))

    def test_all_floats(self):
        """Test return for all float arguments"""

        inputs = 1, 2
        self.assert_return_matching_dimensions(inputs)

    def test_all_1d(self):
        """Test return for all 1D arguments"""

        inputs = [1], [2]
        self.assert_return_matching_dimensions(inputs)

    def test_all_2d(self):
        """Test return for all 2D arguments"""

        inputs = [[1, 2], [1, 2]], [[5, 6], [7, 8]]
        self.assert_return_matching_dimensions(inputs)

    def test_float_1d(self):
        """Test return for mixed float and 1D arguments"""

        inputs = 1, [2, 3]
        expected = [[1, 1], [2, 3]]
        self.assert_return(inputs, expected)

    def test_float_2d(self):
        """Test return for mixed float and 2D arguments"""

        inputs = 1, [[2, 3], [4, 5]]
        expected = [np.ones((2, 2)), inputs[1]]
        self.assert_return(inputs, expected)

    def test_1d_2d(self):
        """Test return for mixed 1D and 2D arguments"""

        inputs = [1, 2], [[2, 3], [4, 5]]
        expected = [[[1, 2], [1, 2]], inputs[1]]
        self.assert_return(inputs, expected)

    def test_error_on_mismatched_dimensions(self):
        """Test an error is raised when input arguments cannot be cast to match dimensions"""

        with self.assertRaises(ValueError):
            CosmologyAccessor._match_argument_dimensions([1, 2], [1, 2, 3])
