# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``reference`` module"""

from unittest import TestCase

import numpy as np
from astropy.table import Table
from pandas.testing import assert_series_equal

from sn_analysis import reference


class StellarSpectraParsing(TestCase):
    """Test ``get_stellar_spectra`` returns a spectrum that is the same as
    directly parsing the file of a stellar type using
    ``_read_stellar_spectra_path``
    """

    @classmethod
    def setUpClass(cls):
        cls.file_names = [
            'F5_lte07000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'G2_lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'K2_lte04900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'K5_lte04400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'K9_lte04100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M0_lte03800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M1_lte03600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M2_lte03400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M3_lte03200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M4_lte03100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M5_lte02800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            'M9_lte02300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        ]

        cls.stellar_types = [f[:2] for f in cls.file_names]

    def runTest(self):
        for stellar_type, fname in zip(self.stellar_types, self.file_names):
            full_path = reference._STELLAR_SPECTRA_DIR / fname
            spec_by_path = reference._read_stellar_spectra_path(full_path)
            spec_by_type = reference.get_stellar_spectra(stellar_type)
            assert_series_equal(spec_by_path, spec_by_type)


class GetReferenceStarDataframe(TestCase):
    """Tests for the ``get_ref_star_dataframe`` function"""

    @classmethod
    def setUpClass(cls):
        cls.ref_star_dataframe = reference.get_ref_star_dataframe()

    def test_includes_unnormalized_flux(self):
        """Tests band flux columns are included in the dataframe"""

        self.assertTrue(
            [c for c in self.ref_star_dataframe.columns if not c.endswith('_norm')]
        )

    def test_includes_normalized_flux(self):
        """Tests normalized flux columns are included in the dataframe"""

        self.assertTrue(
            [c for c in self.ref_star_dataframe.columns if c.endswith('_norm')]
        )

    def test_flux_normalization_pwv_0(self):
        """Tests normalized flux values are one for PWV = 0"""

        norm_cols = [c for c in self.ref_star_dataframe.columns if 'norm' in c]
        reference_flux = self.ref_star_dataframe.loc[0][norm_cols]

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

        for stellar_type in reference.available_types:
            dataframe = reference.get_ref_star_dataframe(stellar_type)
            self.assertFalse(dataframe.empty)


class InterpNormFlux(TestCase):
    """Tests for the ``interp_norm_flux`` function"""

    test_band = 'lsst_hardware_z'

    def test_norm_flux_is_1_at_zero_pwv(self):
        """Test flux is 1 at the PWV=0 in the test band"""

        norm_flux = reference.interp_norm_flux(self.test_band, pwv=0)
        self.assertEqual(1, norm_flux)

    def test_pwv_is_float_return_is_float(self):
        """Test return is a float when pwv arg is a float"""

        returned_flux = reference.interp_norm_flux(self.test_band, 5)
        self.assertIsInstance(returned_flux, float)

    def test_pwv_is_array_return_is_array(self):
        """Test return is an array when pwv arg is an array"""

        n1d_flux = reference.interp_norm_flux(self.test_band, [5, 6])
        self.assertIsInstance(n1d_flux, np.ndarray)
        self.assertEqual(1, np.ndim(n1d_flux))

    def test_error_out_of_bound(self):
        """Test a value error is raise if PWV is out of range"""

        self.assertRaises(ValueError, reference.interp_norm_flux, self.test_band, 100)


class DivideRefFromLc(TestCase):
    """Tests for the ``divide_ref_from_lc`` function"""

    def setUp(self):
        # Create a dummy table. We don't care that the flux values
        # are non-physical for this set of tests
        self.test_table = Table()
        self.test_table['flux'] = np.arange(10, 26, dtype='float')
        self.test_table['band'] = 'lsst_hardware_z'
        self.test_table['zp'] = 25
        self.test_table['zpsys'] = 'ab'

    def test_no_argument_mutation(self):
        """Test argument table is not mutated"""

        original_table = self.test_table.copy()
        reference.divide_ref_from_lc(self.test_table, pwv=15)
        self.assertTrue(all(original_table == self.test_table))

    def test_returned_flux_is_scaled(self):
        """Test returned table has scaled flux"""

        # Scale the flux values manually
        test_pwv = 15
        test_band = self.test_table['band'][0]
        scale_factor = reference.average_norm_flux(test_band, test_pwv)
        expected_flux = list(self.test_table['flux'] / scale_factor)

        # Scale the flux values with ``divide_ref_from_lc`` and check they
        # match manual results
        scaled_table = reference.divide_ref_from_lc(self.test_table, pwv=test_pwv)
        returned_flux = list(scaled_table['flux'])
        self.assertListEqual(expected_flux, returned_flux)
