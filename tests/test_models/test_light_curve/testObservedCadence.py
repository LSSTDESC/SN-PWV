"""Tests for the ``snat_sim.models.light_curve.ObservedCadence`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.models import ObservedCadence


class SncosmoFormatting(TestCase):
    """Tests the ``to_astropy`` method enforces the sncosmo data model"""

    def setUp(self) -> None:
        """"Create an example cadence and export the data to sncosmo format"""

        observation_times = np.arange(-5, 5)
        self.cadence = ObservedCadence(
            observation_times,
            bands=np.full_like(observation_times, 'sdssr', dtype='U5'),
            zp=25,
            zpsys='AB',
            skynoise=0,
            gain=1
        )

        self.sncosmo_cadence = self.cadence.to_sncosmo()

    def test_correct_table_values(self) -> None:
        """Test the correct zero point and zero-point system were used"""

        np.testing.assert_array_equal(self.sncosmo_cadence['time'], self.cadence.obs_times,'Incorrect observation times')
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
