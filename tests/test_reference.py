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

        returned_mag = reference.ref_star_mag(band, [0])[0]
        self.assertEqual(0, returned_mag)

    def test_zero_mag_for_fiducial_pwv(self):
        """Assert magnitude is zero at fiducial PWV for all bands"""

        for band in ('decam_r', 'decam_i', 'decam_z'):
            self.assertZeroMagBand(band)
