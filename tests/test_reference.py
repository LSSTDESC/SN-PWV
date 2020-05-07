# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``reference`` module"""

from unittest import TestCase

import numpy as np

from sn_analysis import reference


class DefaultPWVConfigVals(TestCase):
    """Tests for the loading of default config values"""

    @classmethod
    def setUpClass(cls):
        """Load the default config values"""
        cls.config_dict = reference.get_config_pwv_vals()

    def test_expected_keys(self):
        """Test returned dictionary has expected keys"""

        returned_keys = set(self.config_dict.keys())
        expected = {'reference_pwv', 'slope_start', 'slope_end'}
        self.assertSequenceEqual(expected, returned_keys)

    def test_values_are_equidistant(self):
        """Test slope start / end values are equidistance from reference PWV"""

        upper_dist = self.config_dict['reference_pwv'] - self.config_dict['slope_end']
        lower_dist = self.config_dict['reference_pwv'] - self.config_dict['slope_start']
        self.assertEqual(upper_dist, -lower_dist)


class ReferenceStarParsing(TestCase):
    """Tests for the loading / parsing of reference star flux values"""

    @classmethod
    def setUpClass(cls):
        cls.ref_star_dataframe = reference.get_ref_star_dataframe()

    def test_includes_base_flux(self):
        """Tests band flux columns are included in the dataframe"""

        test_columns = ['decam_z_flux', 'decam_i_flux', 'decam_r_flux']
        isin = np.isin(test_columns, self.ref_star_dataframe.columns)
        self.assertTrue(isin.all())

    def test_includes_normalized_flux(self):
        """Tests normalized flux columns are included in the dataframe"""

        test_columns = ['decam_z_norm', 'decam_i_norm', 'decam_r_norm']
        isin = np.isin(test_columns, self.ref_star_dataframe.columns)
        self.assertTrue(isin.all())

    def test_flux_normalization(self):
        """Tests normalized flux values are one for reference PWV"""

        config_pwv = reference.get_config_pwv_vals()['reference_pwv']
        norm_cols = [c for c in self.ref_star_dataframe.columns if 'norm' in c]
        reference_flux = self.ref_star_dataframe.loc[config_pwv][norm_cols]

        ones = np.ones_like(reference_flux).tolist()
        self.assertSequenceEqual(ones, list(reference_flux))

    def test_pwv_is_index(self):
        """Test PWV values are set as the index"""

        self.assertEqual('PWV', self.ref_star_dataframe.index.name)

    def test_unknown_stellar_type(self):
        """Test unknown stellar types raise a value error"""

        with self.assertRaises(ValueError):
            reference.get_ref_star_dataframe('a_made_up_stellar_type')

    def test_known_types_parsed(self):
        """Test all stellar types in ``_stellar_type_paths`` are parsed"""

        for stellar_type in reference._stellar_type_paths.keys():
            dataframe = reference.get_ref_star_dataframe(stellar_type)
            self.assertFalse(dataframe.empty)


class ReferenceStarMag(TestCase):
    """Tests for the ``ref_star_mag`` function"""

    def assertZeroMagBand(self, band):
        """Assert magnitude is zero at fiducial PWV in the given band"""

        config_pwv = reference.get_config_pwv_vals()['reference_pwv']
        returned_mag = reference.ref_star_mag(band, [config_pwv])[0]
        self.assertEqual(0, returned_mag)

    def test_zero_mag_for_fiducial_pwv(self):
        """Assert magnitude is zero at fiducial PWV for all bands"""

        for band in ('decam_r', 'decam_i', 'decam_z'):
            self.assertZeroMagBand(band)


class SubtractRefStar(TestCase):
    """Tests for subtracting off reference star magnitudes"""

    def test_recovers_mstar_mag(self):
        """Subtracting the reference star from an array of zero magnitudes
        should return negative the reference flux"""

        test_pwv = [1, 2, 3]
        test_band = 'decam_z'

        ref_mag = reference.ref_star_mag(test_band, test_pwv, 'M9').tolist()
        zeros = np.zeros_like(ref_mag)

        subtracted_mag = reference._subtract_ref_star(test_band, zeros, test_pwv, 'M9')
        self.assertSequenceEqual(ref_mag, list(-subtracted_mag))


class SubtractRefStarSlope(TestCase):
    """Tests for subtracting off reference star slopes"""

    def test_recovers_mstar_slope(self):
        """Subtracting the reference star from an array of zeros
        should return negative the reference slope"""

        test_band = 'decam_z'
        slope_start_pwv = 2
        slope_end_pwv = 6

        # Calculate expected slope
        mag_slope_start, mag_slope_end = reference.ref_star_mag(
            test_band, [slope_start_pwv, slope_end_pwv])

        expected_slope = (mag_slope_end - mag_slope_start) / (slope_end_pwv - slope_start_pwv)

        # Get returned slope
        pwv_config = {'slope_start': slope_start_pwv, 'slope_end': slope_end_pwv}
        returned_slope = reference._subtract_ref_star_slope(
            test_band, [0], pwv_config
        )[0]

        self.assertEqual(expected_slope, -returned_slope)
