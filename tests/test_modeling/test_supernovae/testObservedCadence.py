"""Tests for the ``ObservedCadence`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.models.supernova import ObservedCadence
from ...mock import create_mock_plasticc_light_curve


class SncosmoFormatting(TestCase):

    def setUp(self) -> None:
        obs_times = np.arange(-5, 5)

        self.cadence = ObservedCadence(
            obs_times,
            bands=np.full_like(obs_times, 'sdssr', dtype='U5'),
            zp=25,
            zpsys='AB',
            skynoise=0,
            gain=1
        )

        self.sncosmo_cadence = self.cadence.to_sncosmo()

    def test_correct_table_values(self) -> None:
        """Test the correct zero point and zero-point system were used"""

        np.testing.assert_array_equal(self.sncosmo_cadence['time'], self.cadence.obs_times,
                                      'Incorrect observation times')
        np.testing.assert_array_equal(self.sncosmo_cadence['band'], self.cadence.bands, 'Incorrect band names')
        np.testing.assert_array_equal(self.sncosmo_cadence['zp'], self.cadence.zp, 'Incorrect zero point')
        np.testing.assert_array_equal(self.sncosmo_cadence['zpsys'], self.cadence.zpsys, 'Incorrect zero point system')
        np.testing.assert_array_equal(self.sncosmo_cadence['gain'], self.cadence.gain, 'Incorrect gain')
        np.testing.assert_array_equal(self.sncosmo_cadence['skynoise'], self.cadence.skynoise, 'Incorrect sky noise')

    def test_table_data_types(self) -> None:
        """Test table columns have the expected data types"""

        expected_dtype = np.dtype([
            ('time', '<f8'),
            ('band', '<U1000'),
            ('gain', '<f8'),
            ('skynoise', '<f8'),
            ('zp', '<f8'),
            ('zpsys', '<U100')])

        self.assertEqual(expected_dtype, self.sncosmo_cadence.dtype)


class ExtractCadenceData(TestCase):
    """Tests for the ``extract_cadence_data`` function"""

    def setUp(self) -> None:
        self.plasticc_lc = create_mock_plasticc_light_curve()
        _, self.extracted_cadence = ObservedCadence.from_plasticc(self.plasticc_lc)

    def test_zp_is_overwritten(self) -> None:
        """Test the zero_point in the returned table is overwritten with a constant"""

        np.testing.assert_array_equal(self.plasticc_lc['ZEROPT'], self.extracted_cadence.zp)

    def test_filter_names_are_formatted(self) -> None:
        """Test filter names are formatted for use with sncosmo"""

        is_lower = all(f.islower() for f in self.extracted_cadence.bands)
        self.assertTrue(is_lower, 'Filter names include uppercase letters')

        is_prefixed = all(f.startswith('lsst_hardware_') for f in self.extracted_cadence.bands)
        self.assertTrue(is_prefixed, 'Filter names do not start with ``lsst_hardware_``')

    def test_drop_nondetection(self) -> None:
        """Test ``drop_nondetection=True`` removes non detections"""

        params, extracted_cadence = ObservedCadence.from_plasticc(self.plasticc_lc, drop_nondetection=True)
        returned_dates = extracted_cadence.obs_times
        expected_dates = self.plasticc_lc[self.plasticc_lc['PHOTFLAG'] != 0]['MJD']
        np.testing.assert_array_equal(returned_dates, expected_dates)
