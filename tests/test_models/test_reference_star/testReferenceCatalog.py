"""Tests for the ``snat_sim.models.reference_star.ReferenceCatalog`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.models import ReferenceCatalog, ReferenceStar
from tests.mock import create_mock_light_curve


class InitErrors(TestCase):
    """Test for the raising of appropriate errors at instantiation"""

    def test_value_error_on_missing_args(self) -> None:
        """Test a descriptive ``ValueError`` is raised for missing arguments"""

        with self.assertRaisesRegex(ValueError, 'Must specify at least one spectral type for the catalog.'):
            ReferenceCatalog()


class AverageNormFlux(TestCase):
    """Tests for the ``average_norm_flux`` function"""

    test_band = 'lsst_hardware_z'

    def test_average_matches_ref_stars_for_float(self) -> None:
        """Test the return matches the average norm flux at a single PWV for two reference types"""

        test_pwv = 5
        avg_flux = ReferenceCatalog('G2', 'M5').average_norm_flux(self.test_band, test_pwv)
        g2_flux = ReferenceStar('G2').norm_flux(self.test_band, test_pwv)
        m5_flux = ReferenceStar('M5').norm_flux(self.test_band, test_pwv)
        self.assertEqual(avg_flux, np.average((g2_flux, m5_flux)))

    def test_average_matches_ref_star_for_array(self) -> None:
        """Test the return matches the average norm flux at an array of PWV for two reference types"""

        test_pwv = [5, 6]
        avg_flux = ReferenceCatalog('G2', 'M5').average_norm_flux(self.test_band, test_pwv)
        g2_flux = ReferenceStar('G2').norm_flux(self.test_band, test_pwv)
        m5_flux = ReferenceStar('M5').norm_flux(self.test_band, test_pwv)

        self.assertIsInstance(avg_flux, np.ndarray, 'Returned average was not an array')
        np.testing.assert_array_equal(avg_flux, np.average((g2_flux, m5_flux), axis=0))


class CalibrateLc(TestCase):
    """Tests for the ``calibrate_lc`` function"""

    def setUp(self) -> None:
        """Create a mock light-curve and a stellar reference catalog"""

        self.light_curve = create_mock_light_curve()
        self.catalog = ReferenceCatalog('G2', 'M5')

    def test_no_argument_mutation(self) -> None:
        """Test argument table is not mutated"""

        original_data = self.light_curve.copy()
        self.catalog.calibrate_lc(self.light_curve, pwv=15)
        self.assertTrue(original_data == self.light_curve)

    def test_flux_is_scaled_for_pwv_float(self) -> None:
        """Test flux values are scaled according to a scalar PWV value"""

        pwv = 15

        # Scale the flux values manually
        scale_factor = [self.catalog.average_norm_flux(b, pwv) for b in self.light_curve.band]
        expected_flux = np.divide(self.light_curve.flux, scale_factor)

        # Scale the flux values with ``calibrate_lc`` and check they match manual results
        scaled_table = self.catalog.calibrate_lc(self.light_curve, pwv=pwv)
        np.testing.assert_array_equal(expected_flux, scaled_table.flux)

    def test_flux_is_scaled_for_pwv_vector(self) -> None:
        """Test flux values are scaled according to a vector of PWV values"""

        pwv = np.full(len(self.light_curve), 15)

        # Scale the flux values manually
        scale_factor = [self.catalog.average_norm_flux(b, p) for b, p in zip(self.light_curve.band, pwv)]
        expected_flux = np.divide(self.light_curve.flux, scale_factor)

        # Scale the flux values with ``calibrate_lc`` and check they match manual results
        scaled_table = self.catalog.calibrate_lc(self.light_curve, pwv=pwv)
        np.testing.assert_array_equal(expected_flux, scaled_table.flux)

    def test_error_on_mismatched_arg_length(self) -> None:
        """Test a value error is raised when argument lengths are not the same"""

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            self.catalog.calibrate_lc(self.light_curve, pwv=[1])
