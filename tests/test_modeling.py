# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``modeling`` module"""

import types
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from pwv_kpno import pwv_atm

from sn_analysis import modeling
from sn_analysis.utils import register_decam_filters

register_decam_filters(force=True)


class TestTransmissionEffects(TestCase):
    """Tests for the addition of PWV to sncosmo models"""

    def setUp(self):
        self.pwv = 10
        self.z = .8
        self.model = modeling.get_model_with_pwv('salt2-extended')

    def test_recovered_transmission(self):
        """The simulated SN flux with PWV / the flux without PWV should be
        equivalent to the PWV transmission function
        """

        wavelengths = np.arange(4000, 10000)

        self.model.set(pwv=0, z=self.z)
        flux = self.model.flux(0, wavelengths)

        self.model.set(pwv=self.pwv, z=self.z)
        flux_pwv = self.model.flux(0, wavelengths)

        transmission = pwv_atm.trans_for_pwv(self.pwv)
        interp_transmission = np.interp(
            wavelengths,
            transmission['wavelength'],
            transmission['transmission'])

        is_close = np.isclose(interp_transmission, flux_pwv / flux).all()
        if not is_close:
            plt.plot(wavelengths, flux_pwv / flux)
            plt.title('Incorrect Recovered Transmission')
            plt.xlabel('Wavelength')
            plt.ylabel('(Flux PWV=10) / (Flux PWV=0)')
            plt.show()

        self.assertTrue(is_close)


class TestObservationsTableCreation(TestCase):
    """Tests for creation of the observations (cadence) table"""

    @classmethod
    def setUpClass(cls):
        cls.phases = range(-15, 40)
        cls.bands = ('decam_g', 'decam_r', 'decam_i', 'decam_z')
        cls.zp = 15
        cls.zpsys = 'SDSS'

        cls.observations_table = modeling.create_observations_table(
            phases=cls.phases,
            bands=cls.bands,
            zp=cls.zp,
            zpsys=cls.zpsys
        )

    def test_correct_zero_point(self):
        """Test the correct zero point and zeropoint system were used"""

        self.assertTrue(
            all(self.observations_table['zp'] == self.zp),
            'Incorrect zero point'
        )

        self.assertTrue(
            all(self.observations_table['zpsys'] == self.zpsys),
            'Incorrect zero point system'
        )

    def test_observation_table_bands(self):
        """Test each band is specified for every single day"""

        dataframe = self.observations_table.to_pandas()
        counts = dataframe.band.value_counts()
        for band in self.bands:
            self.assertEqual(len(self.phases), counts[band])

    def test_phase_range(self):
        """Test each specified phase is in the returned table"""

        self.assertEqual(
            set(self.phases), set(self.observations_table['time']))


class TestLCSimulation(TestCase):
    """Tests for light-curve simulation"""

    def setUp(self):
        """Create a new light-curve iterator for each test"""

        self.pwv_vals = 0, 5
        self.z_vals = 0, 1
        source = 'salt2-extended'
        self.observations = modeling.create_observations_table()
        self.lc_iter = modeling.iter_lcs(
            self.observations,
            source,
            self.pwv_vals,
            self.z_vals,
            verbose=False)

    def test_correct_meta_data(self):
        """Test light-curve meta data reflects expected simulation params"""

        for pwv in self.pwv_vals:
            for z in self.z_vals:
                x0 = modeling.calc_x0_for_z(z)
                expected_meta = {'t0': 0, 'pwv': pwv, 'z': z, 'x0': x0}
                self.assertEqual(expected_meta, next(self.lc_iter).meta)

    def assertSimValuesEqualObs(self, key, lc=None):
        """Assert values in the observation table match values in the
        light-curve table.

        Args:
            key  (str): Name of the column to test
            lc (Table): Light-curve table to test against observations
        """

        if not lc:
            lc = next(self.lc_iter)

        obs_times = self.observations[key]
        sim_times = lc[key]
        self.assertSequenceEqual(list(obs_times), list(sim_times))

    def test_obs_table_matches_simulation(self):
        """Test band and time values in the simulated light-curves match the
         observations table """

        self.assertSimValuesEqualObs('time')
        self.assertSimValuesEqualObs('band')

    def test_return_is_iter(self):
        """Test the returned light-curve collection is an generator"""

        self.assertIsInstance(self.lc_iter, types.GeneratorType)


class calc_x0_for_z(TestCase):
    """Tests for the ``scale_model_to_redshift`` function"""

    def test_z_is_zero(self):
        """Test x_0 == 1 when z == 0"""

        self.assertEqual(1, modeling.calc_x0_for_z(0))

    def test_z_approx_zero(self):
        """Test x_0 is approximately 1 when z is approximately 0

        Tests a different path than ``test_z_is_zero``
        """

        self.assertEqual(1, modeling.calc_x0_for_z(1E-500))
