"""Tests for the ``CosmologyAccessor`` class"""

from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd

from snat_sim import constants as const
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
