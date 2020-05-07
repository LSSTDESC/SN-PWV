# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``sn_magnitudes`` module"""

from unittest import TestCase

import numpy as np

from sn_analysis import modeling, sn_magnitudes
from utils import register_decam_filters

register_decam_filters(force=True)


class TestTabulateMagnitudes(TestCase):
    """Tests for the ``tabulate_mag`` function"""

    @classmethod
    def setUpClass(cls):
        cls.source = 'salt2-extended'
        cls.pwv_vals = 0, 5
        cls.z_vals = 0, .5
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
                model.set(pwv=pwv, z=z)
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


class TestTabulateFiducialMag(TestCase):
    """Tests for the ``tabulate_fiducial_mag`` function"""

    @classmethod
    def setUpClass(cls):
        cls.source = 'salt2-extended'
        cls.z_vals = [0, .5]
        cls.bands = ['sdssu', 'sdssg']
        cls.fiducial_pwv_dict = {
            'reference_pwv': 5,
            'slope_start': 6,
            'slope_end': 7
        }

    def get_mag_for_pwv(self, key):
        """Return magnitude for pwv specified in ``self.fiducial_pwv_dict``

        Args:
            key (str): Key in fiducial pwv dictionary

        Returns:
            An array of magnitudes for each redshift in ``self.z_vals``
        """

        model = modeling.get_model_with_pwv(self.source)
        pwv = self.fiducial_pwv_dict[key]

        mags = []
        for z in self.z_vals:
            model.set(z=z, pwv=pwv)
            mags.append(model.bandmag(self.band, 'ab', 0))

        return mags

    def test_values_match_simulation(self):
        """Test returned values equal simulated values"""

        fiducial_mag = sn_magnitudes.tabulate_fiducial_mag(
            self.source, self.z_vals, self.band, self.fiducial_pwv_dict)

        for key in self.fiducial_pwv_dict.keys():
            mag = self.get_mag_for_pwv('reference_pwv')
            self.assertTrue(np.isclose(mag, fiducial_mag[key]).all())

    def test_returned_array_shape(self):
        """Test returned dictionary values are arrays with correct shape"""

        for band, array in self.mag_dict.items():
            self.assertEqual(shape, array.shape, f'Wrong shape for {band}')

