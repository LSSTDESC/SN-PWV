"""Tests for the ``FixedResTransmission`` class"""

from unittest import TestCase

import numpy as np
import pandas as pd

from snat_sim import models


class PWVInterpolation(TestCase):
    """Test the interpolation of PWV values from the underlying data"""

    def setUp(self):
        """Create a dummy ``Transmission`` object"""

        self.transmission = models.FixedResTransmission(4)

    def test_interpolation_on_grid_point(self):
        """Test interpolation result matches sampled values at the grid points"""

        test_pwv = self.transmission.samp_pwv[1]
        expected_transmission = self.transmission.samp_transmission[1]

        returned_trans = self.transmission.calc_transmission(test_pwv)
        np.testing.assert_equal(expected_transmission, returned_trans)

    def test_interpolates_for_given_wavelengths(self):
        """Test an interpolation is performed for specified wavelengths when given"""

        test_pwv = self.transmission.samp_pwv[1]
        test_wave = np.arange(3000, 3500, 50)

        returned_wave = self.transmission.calc_transmission(test_pwv, wave=test_wave).index.values
        np.testing.assert_equal(returned_wave, test_wave)


class FunctionDefaults(TestCase):
    """Test the transmission evaluations default to values specified at innit"""

    def setUp(self):
        """Create a dummy ``Transmission`` object"""

        self.transmission = models.FixedResTransmission(4)

    def test_default_wavelengths_match_sampled_wavelengths(self):
        """Test return values are index by sample wavelengths by default"""

        np.testing.assert_equal(self.transmission.calc_transmission(4).index.to_numpy(), self.transmission.samp_wave)


class VectorPWVSupport(TestCase):
    """Test the handling of list-like PWV arguments"""

    def setUp(self):
        """Create a dummy ``Transmission`` object"""

        self.transmission = models.FixedResTransmission(4)

    def test_scalar_pwv_returns_series(self):
        """Test passing a scalar PWV value returns a pandas Series object"""

        transmission = self.transmission.calc_transmission(4)
        self.assertIsInstance(transmission, pd.Series)
        self.assertEqual(transmission.name, f'4.0 mm')

    def test_vector_pwv_returns_dataframe(self):
        """Test passing a vector of PWV values returns a pandas DataFrame"""

        transmission = self.transmission.calc_transmission([4, 5])
        self.assertIsInstance(transmission, pd.DataFrame)
        np.testing.assert_equal(transmission.columns.values, [f'4.0 mm', f'5.0 mm'])
