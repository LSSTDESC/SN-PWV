"""Tests for the ``ReferenceStar`` class"""

from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from snat_sim._data_paths import data_paths
from snat_sim.reference_stars import ReferenceStar


class InitErrors(TestCase):
    """Test for the raising of errors at instantiation"""

    def test_value_error_on_missing_data(self) -> None:
        """Test for value error on bad spectral type"""

        with self.assertRaises(ValueError):
            ReferenceStar('DummySpectralType')

    def test_case_insensitive(self) -> None:
        """Test the spectral type argument is case insensitive"""

        uppercase = ReferenceStar('G2')
        lowercase = ReferenceStar('g2')

        # Test the same spectrum is loaded
        pd.testing.assert_series_equal(uppercase.to_pandas(), lowercase.to_pandas())
        self.assertEqual('G2', lowercase.spectral_type)


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

    def runTest(self) -> None:
        for stellar_type, fname in zip(self.stellar_types, self.file_names):
            full_path = data_paths.stellar_spectra_dir / fname
            spec_by_path = ReferenceStar._read_stellar_spectra_path(full_path)
            spec_by_type = ReferenceStar(stellar_type).to_pandas()
            assert_series_equal(spec_by_path, spec_by_type)


class GetReferenceStarDataframe(TestCase):
    """Tests for the ``get_dataframe`` function"""

    @classmethod
    def setUpClass(cls):
        cls.ref_star_dataframe = ReferenceStar('G2').get_dataframe()

    def test_includes_unnormalized_flux(self) -> None:
        """Tests band flux columns are included in the dataframe"""

        self.assertTrue(
            [c for c in self.ref_star_dataframe.columns if not c.endswith('_norm')]
        )

    def test_includes_normalized_flux(self) -> None:
        """Tests normalized flux columns are included in the dataframe"""

        self.assertTrue(
            [c for c in self.ref_star_dataframe.columns if c.endswith('_norm')]
        )

    def test_flux_normalization_pwv_0(self) -> None:
        """Tests normalized flux values are one for PWV = 0"""

        norm_cols = [c for c in self.ref_star_dataframe.columns if 'norm' in c]
        reference_flux = self.ref_star_dataframe.loc[0][norm_cols]

        ones = np.ones_like(reference_flux).tolist()
        self.assertSequenceEqual(ones, list(reference_flux))

    def test_pwv_is_index(self) -> None:
        """Test PWV values are set as the index"""

        self.assertEqual('PWV', self.ref_star_dataframe.index.name)


class InterpNormFlux(TestCase):
    """Tests for the flux calculation"""

    @classmethod
    def setUpClass(cls):
        cls.ref_star = ReferenceStar('G2')
        cls.test_band = 'lsst_hardware_z'

    def test_pwv_is_float_return_is_float(self) -> None:
        """Test return is a float when pwv arg is a float"""

        returned_flux = self.ref_star.flux(self.test_band, 5)
        self.assertIsInstance(returned_flux, float)

    def test_pwv_is_array_return_is_array(self) -> None:
        """Test return is an array when pwv arg is an array"""

        n1d_flux = self.ref_star.flux(self.test_band, [5, 6])
        self.assertIsInstance(n1d_flux, np.ndarray)
        self.assertEqual(1, np.ndim(n1d_flux))

    def test_error_out_of_bound(self) -> None:
        """Test a value error is raise if PWV is out of range"""

        self.assertRaises(ValueError, self.ref_star.flux, self.test_band, 100)
        self.assertRaises(ValueError, self.ref_star.flux, self.test_band, -1)


class NormFlux(TestCase):
    """Tests for the calculation of the normalized flux"""

    @classmethod
    def setUpClass(cls):
        cls.ref_star = ReferenceStar('G2')
        cls.test_band = 'lsst_hardware_z'

    def test_norm_flux_is_1_at_zero_pwv(self) -> None:
        """Test flux is 1 at the PWV=0 in the test band"""

        norm_flux = self.ref_star.norm_flux(self.test_band, pwv=0)
        self.assertEqual(1, norm_flux)

    def test_pwv_is_float_return_is_float(self) -> None:
        """Test return is a float when pwv arg is a float"""

        returned_flux = self.ref_star.norm_flux(self.test_band, 5)
        self.assertIsInstance(returned_flux, float)

    def test_pwv_is_array_return_is_array(self) -> None:
        """Test return is an array when pwv arg is an array"""

        n1d_flux = self.ref_star.norm_flux(self.test_band, [5, 6])
        self.assertIsInstance(n1d_flux, np.ndarray)
        self.assertEqual(1, np.ndim(n1d_flux))

    def test_error_out_of_bound(self) -> None:
        """Test a value error is raise if PWV is out of range"""

        self.assertRaises(ValueError, self.ref_star.norm_flux, self.test_band, 100)
        self.assertRaises(ValueError, self.ref_star.norm_flux, self.test_band, -1)
