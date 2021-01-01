"""Tests for the ``ReferenceCatalog`` class"""

from unittest import TestCase

import numpy as np
from astropy.table import Table

from snat_sim.reference_stars import ReferenceStar, ReferenceCatalog


class AverageNormFlux(TestCase):
    """Tests for the ``average_norm_flux`` function"""

    test_band = 'lsst_hardware_z'

    def test_average_matches_ref_stars_for_float(self):
        """Test the return matches the average norm flux at a single PWV for two reference types"""

        test_pwv = 5
        avg_flux = ReferenceCatalog('G2', 'M5').average_norm_flux(self.test_band, test_pwv)
        g2_flux = ReferenceStar('G2').norm_flux(self.test_band, test_pwv)
        m5_flux = ReferenceStar('M5').norm_flux(self.test_band, test_pwv)
        self.assertEqual(avg_flux, np.average((g2_flux, m5_flux)))

    def test_average_matches_ref_star_for_array(self):
        """Test the return matches the average norm flux at an array of PWV for two reference types"""

        test_pwv = [5, 6]
        avg_flux = ReferenceCatalog('G2', 'M5').average_norm_flux(self.test_band, test_pwv)
        g2_flux = ReferenceStar('G2').norm_flux(self.test_band, test_pwv)
        m5_flux = ReferenceStar('M5').norm_flux(self.test_band, test_pwv)

        self.assertIsInstance(avg_flux, np.ndarray, 'Returned average was not an array')
        np.testing.assert_array_equal(avg_flux, np.average((g2_flux, m5_flux), axis=0))


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

        self.catalog = ReferenceCatalog('G2', 'M5')

    def test_no_argument_mutation(self):
        """Test argument table is not mutated"""

        original_table = self.test_table.copy()
        self.catalog.divide_ref_from_lc(self.test_table, pwv=15)
        self.assertTrue(all(original_table == self.test_table))

    def assert_returned_flux_is_scaled(self, pwv):
        """Test returned table has scaled flux"""

        # Scale the flux values manually
        test_band = self.test_table['band'][0]
        scale_factor = self.catalog.average_norm_flux(test_band, pwv)
        expected_flux = np.divide(self.test_table['flux'], scale_factor)

        # Scale the flux values with ``divide_ref_from_lc`` and check they
        # match manual results
        scaled_table = self.catalog.divide_ref_from_lc(self.test_table, pwv=pwv)
        returned_flux = list(scaled_table['flux'])
        np.testing.assert_array_equal(expected_flux, returned_flux)

    def test_flux_is_scaled_for_pwv_float(self):
        """Test flux values are scaled according to a scalar PWV value"""

        self.assert_returned_flux_is_scaled(pwv=15)

    def test_flux_is_scaled_for_pwv_vector(self):
        """Test flux values are scaled according to a vector of PWV values"""

        pwv_array = np.full(len(self.test_table), 15)
        self.assert_returned_flux_is_scaled(pwv=pwv_array)
        self.assert_returned_flux_is_scaled(pwv=pwv_array.tolist())

    def test_error_on_mismatched_arg_length(self):
        """Test a value error is raised when argument lengths are not the same"""

        with self.assertRaises(ValueError):
            self.catalog.divide_ref_from_lc(self.test_table, pwv=[1])
