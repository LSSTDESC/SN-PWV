# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``sn_magnitudes`` module"""

from unittest import TestCase

import numpy as np

from sn_analysis import modeling, sn_magnitudes
from sn_analysis.filters import register_decam_filters

register_decam_filters(force=True)


class TestTabulateMagnitudes(TestCase):
    """Tests for the ``tabulate_mag`` function"""

    @classmethod
    def setUpClass(cls):
        cls.source = 'salt2-extended'
        cls.pwv_vals = 0.001, 5
        cls.z_vals = 0.001, .5
        cls.bands = ['sdssu', 'sdssg']

        cls.mag_dict = sn_magnitudes.tabulate_mag(
            cls.source, cls.pwv_vals, cls.z_vals, cls.bands)

    def test_values_match_sncosmo_simulation(self):
        """Test returned values equal simulated values from sncosmo"""

        # Tabulated results for a single band
        test_band = self.bands[0]
        tabulated_mag = self.mag_dict[test_band]

        # Determine magnitude values independently
        model = modeling.get_model_with_pwv(self.source)
        expected_mag = []
        for i, pwv in enumerate(self.pwv_vals):
            mag_for_pwv = []

            for j, z in enumerate(self.z_vals):
                model.set(pwv=pwv, z=z, x0=modeling.calc_x0_for_z(z, self.source))
                mag = model.bandmag(test_band, 'ab', 0)
                mag_for_pwv.append(mag)

            expected_mag.append(mag_for_pwv)

        np.testing.assert_equal(expected_mag, tabulated_mag)

    def test_all_bands_returned(self):
        """Test values are returned for each input band"""

        returned_bands = list(self.mag_dict.keys())
        self.assertSequenceEqual(self.bands, returned_bands)

    def test_returned_array_shape(self):
        """Test returned dictionary values are arrays with correct shape"""

        shape = len(self.pwv_vals), len(self.z_vals)
        for band, array in self.mag_dict.items():
            self.assertEqual(shape, array.shape, f'Wrong shape for {band}')
