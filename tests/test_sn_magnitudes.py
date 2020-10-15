# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``sn_magnitudes`` module"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim import modeling, sn_magnitudes
from snat_sim.filters import register_decam_filters

register_decam_filters(force=True)


class GetConfigPWVValues(TestCase):
    """Tests for the ``get_config_pwv_vals`` function"""

    @classmethod
    def setUpClass(cls):
        """Load the default config values"""
        cls.config_dict = sn_magnitudes.get_config_pwv_vals()

    def test_expected_keys(self):
        """Test returned dictionary has expected keys"""

        returned_keys = set(self.config_dict.keys())
        expected = {'reference_pwv', 'slope_start', 'slope_end'}
        self.assertSequenceEqual(expected, returned_keys)

    def test_values_are_equidistant(self):
        """Test slope start / end values are equidistant from reference PWV"""

        upper_dist = self.config_dict['reference_pwv'] - self.config_dict['slope_end']
        lower_dist = self.config_dict['reference_pwv'] - self.config_dict['slope_start']
        self.assertEqual(upper_dist, -lower_dist)


class TestTabulateMagnitudes(TestCase):
    """Tests for the ``tabulate_mag`` function"""

    def setUp(self):
        self.model = sncosmo.Model('salt2-extended')
        self.model.add_effect(modeling.StaticPWVTrans(), '', 'obs')

        self.pwv_vals = 0.001, 5
        self.z_vals = 0.001, .5
        self.bands = ['sdssu', 'sdssg']

        self.mag_dict = sn_magnitudes.tabulate_mag(
            self.model, self.pwv_vals, self.z_vals, self.bands)

    def test_values_match_sncosmo_simulation(self):
        """Test returned values equal simulated values from sncosmo"""

        # Tabulated results for a single band
        test_band = self.bands[0]
        tabulated_mag = self.mag_dict[test_band]

        # Determine magnitude values independently
        expected_mag = []
        for i, pwv in enumerate(self.pwv_vals):
            mag_for_pwv = []

            for j, z in enumerate(self.z_vals):
                self.model.set(pwv=pwv, z=z, x0=modeling.calc_x0_for_z(z, self.model.source))
                mag = self.model.bandmag(test_band, 'ab', 0)
                mag_for_pwv.append(mag)

            expected_mag.append(mag_for_pwv)

        np.testing.assert_allclose(expected_mag, tabulated_mag)

    def test_all_bands_returned(self):
        """Test values are returned for each input band"""

        returned_bands = list(self.mag_dict.keys())
        self.assertSequenceEqual(self.bands, returned_bands)

    def test_returned_array_shape(self):
        """Test returned dictionary values are arrays with correct shape"""

        shape = len(self.pwv_vals), len(self.z_vals)
        for band, array in self.mag_dict.items():
            self.assertEqual(shape, array.shape, f'Wrong shape for {band}')
