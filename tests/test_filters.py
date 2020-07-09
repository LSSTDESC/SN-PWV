# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``filters`` module"""

from unittest import TestCase

import numpy as np
import sncosmo

from sn_analysis import filters


class RegisterSncosmoFilter(TestCase):
    """Tests for the ``register_sncosmo_filter``"""

    def setUp(self):
        self.wave = np.arange(1000, 10_000)
        self.transmission = np.full_like(self.wave, 0.75, dtype=float)
        self.name = 'test_band'

        filters.register_sncosmo_filter(
            self.wave, self.transmission, self.name, force=True)

    def test_filter_is_registered(self):
        """Check test filter is registered with correct wavelength and transmission"""

        sncosmo_band = sncosmo.get_bandpass(self.name)
        self.assertListEqual(
            self.wave.tolist(), sncosmo_band.wave.tolist(),
            'Incorrect wavelengths for bandpass')

        self.assertListEqual(
            self.transmission.tolist(), sncosmo_band.trans.tolist(),
            'Incorrect transmission for bandpass')

    def test_error_without_force(self):
        """Test error raised if bandpasses are re-registered with force=False"""

        args = self.wave, self.transmission, self.name
        self.assertRaises(TypeError, filters.register_sncosmo_filter, args)


class RegisterDECAMFilters(TestCase):
    """Tests for the ``register_decam_filters``"""

    @classmethod
    def setUpClass(cls):
        """Register the DECam filters"""

        filters.register_decam_filters(force=True)

    def assert_bands_are_registered(self, *bands):
        """Fail if given bands are not registered with sncosmo"""

        for band in bands:
            sncosmo.get_bandpass(band)

    def test_bandpasses_registered(self):
        """Test bands are registered under ids DECam_<ugrizy>"""

        bands = [f'DECam_{b}' for b in 'ugrizY']
        self.assert_bands_are_registered(*bands)

    def test_fitlers_registered(self):
        """Test bands are registered under ids DECam_<ugrizy>_filter"""

        bands = [f'DECam_{b}_filter' for b in 'ugrizY']
        self.assert_bands_are_registered(*bands)

    def test_ccd_registered(self):
        """Test a ``DECam_ccd`` band is registered"""

        self.assert_bands_are_registered('DECam_ccd')

    def test_throughput_registered(self):
        """Test a ``DECam_atm`` band is registered"""

        self.assert_bands_are_registered('DECam_atm')


class RegisterLSSTFilters(TestCase):
    """Tests for the ``register_lsst_filters``"""

    @classmethod
    def setUpClass(cls):
        """Register the LSST filters"""

        filters.register_lsst_filters(force=True)

    def assert_bands_are_registered(self, *bands):
        """Fail if given bands are not registered with sncosmo"""

        for band in bands:
            sncosmo.get_bandpass(band)

    def test_total_bands_registered(self):
        """Test bands are registered under ids lsst_total_<ugrizy>"""

        bands = [f'lsst_total_{b}' for b in 'ugrizY']
        self.assert_bands_are_registered(*bands)

    def test_hardware_registered(self):
        """Test bands are registered under ids lsst_hardware_<ugrizy>"""

        bands = [f'lsst_hardware_{b}' for b in 'ugrizY']
        self.assert_bands_are_registered(*bands)

    def test_filters_registered(self):
        """Test bands are registered under ids lsst_filter_<ugrizy>"""

        bands = [f'lsst_filter_{b}' for b in 'ugrizY']
        self.assert_bands_are_registered(*bands)

    def test_mirrors_registered(self):
        """Test bands are registered under ids lsst_m<1, 2, 3>"""

        bands = [f'lsst_m{m}' for m in range(1, 4)]
        self.assert_bands_are_registered(*bands)

    def test_lenses_registered(self):
        """Test bands are registered under ids lsst_lens<1, 2, 3>"""

        bands = [f'lsst_lens{m}' for m in range(1, 4)]
        self.assert_bands_are_registered(*bands)

    def test_detector_registered(self):
        """Test a ``lsst_detector`` band is registered"""

        self.assert_bands_are_registered('lsst_detector')

    def test_atmospheres_registered(self):
        """Test the ``atmos_10`` and ``atmos_10_std`` bands are registered"""

        bands = ['lsst_atmos_10', 'lsst_atmos_std']
        self.assert_bands_are_registered(*bands)

    def test_combined_mirrors_equal_product_of_mirrors(self):
        """Test ``lsst_mirrors`` filter is the product of all three mirrors"""

        m1 = sncosmo.get_bandpass('lsst_m1')
        m2 = sncosmo.get_bandpass('lsst_m2')
        m3 = sncosmo.get_bandpass('lsst_m3')
        m_product = m1.trans * m2.trans * m3.trans

        m_total = sncosmo.get_bandpass('lsst_mirrors')
        self.assertListEqual(m_product.tolist(), m_total.trans.tolist())

    def test_combined_lenses_equal_product_of_lenses(self):
        """Test ``lsst_lenses`` filter is the product of all three lenses"""

        l1 = sncosmo.get_bandpass('lsst_lens1')
        l2 = sncosmo.get_bandpass('lsst_lens2')
        l3 = sncosmo.get_bandpass('lsst_lens3')
        l_product = l1.trans * l2.trans * l3.trans

        l_total = sncosmo.get_bandpass('lsst_lenses')
        self.assertListEqual(l_product.tolist(), l_total.trans.tolist())

    def test_hardware_equal_product_of_components(self):
        """Test ``lsst_hardware`` filter is product of all non-Atm. filters"""

        wave = np.arange(3000, 11000)
        mirrors = sncosmo.get_bandpass('lsst_mirrors')
        lenses = sncosmo.get_bandpass('lsst_lenses')
        detector = sncosmo.get_bandpass('lsst_detector')

        for band in 'ugrizy':
            filter = sncosmo.get_bandpass(f'lsst_filter_{band}')
            hardware = sncosmo.get_bandpass(f'lsst_hardware_{band}')
            expected_trans = mirrors(wave) * lenses(wave) * detector(wave) * filter(wave)

            # We use ``all_close`` to allow for differences due to interpolation
            np.testing.assert_allclose(
                expected_trans, hardware(wave), atol=1e-10,
                err_msg=f'Incorrect hardware transmission for {band}')
