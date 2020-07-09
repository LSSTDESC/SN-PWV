# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``modeling`` module"""

import types
from unittest import TestCase, skip

import numpy as np
import sncosmo
from astropy.table import Table

from sn_analysis import modeling
from sn_analysis.filters import register_decam_filters

register_decam_filters(force=True)


class GetModelWithPWV(TestCase):
    """Tests for the ``get_model_with_pwv`` function"""

    def test_model_has_pwv_component(self):
        """Test returned model has a PWV transmission component"""

        model = modeling.get_model_with_pwv('salt2')
        effect_types = [type(eff) for eff in model.effects]
        self.assertIn(modeling.PWVTrans, effect_types)

    def test_kwarg_params_are_set(self):
        """Test kwargs are set as model parameters"""

        test_pwv = 10
        test_resolution = 6
        model = modeling.get_model_with_pwv('salt2', pwv=test_pwv, res=test_resolution)
        self.assertEqual(test_pwv, model['pwv'], 'Model has incorrect PWV value')
        self.assertEqual(test_resolution, model['res'], 'Model has incorrect PWV resolution')


class CalcX0ForZ(TestCase):
    """Tests for the ``calc_x0_for_z`` function"""

    @skip
    def test_x0_recovers_absolute_mag(self):
        """Test returned x0 corresponds to specified magnitude"""

        z = 1.5
        source = 'salt2-extended'
        abs_mag = -18
        band = 'standard::b'
        x0 = modeling.calc_x0_for_z(z, source, abs_mag=abs_mag)

        model = sncosmo.Model(source)
        model.set(z=z, x0=x0)
        recovered_mag = model.source_peakabsmag(band, 'AB')
        self.assertEqual(abs_mag, recovered_mag)


class CreateObservationsTable(TestCase):
    """Tests for the ``create_observations_table`` function"""

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

    def test_table_datatypes(self):
        """Test table columns have the expected data types"""

        expected_dtype = np.dtype([
            ('time', '<f8'),
            ('band', '<U1000'),
            ('gain', '<f8'),
            ('skynoise', '<f8'),
            ('zp', '<f8'),
            ('zpsys', '<U100')])

        self.assertEqual(expected_dtype, self.observations_table.dtype)


class RealizeLC(TestCase):
    """Tests for the ``realize_lc`` function"""

    def setUp(self):
        """Simulate a cadence and associated light-curve"""

        self.observations = modeling.create_observations_table()
        self.source = 'salt2-extended'

        z = 0.5
        self.snr = 12
        self.params = dict(
            pwv=0.01,
            x1=.8,
            c=-.5,
            z=z,
            t0=1,
            x0=modeling.calc_x0_for_z(z, self.source),
            res=5
        )

        self.obs = modeling.create_observations_table()
        self.simulated_lc = modeling.realize_lc(self.obs, self.source, self.snr, **self.params)

    def test_simulated_snr(self):
        """Test SNR of simulated light-curve equals snr kwarg"""

        simulated_snr = self.simulated_lc['flux'] / self.simulated_lc['fluxerr']
        simulated_snr = np.round(simulated_snr, 3).tolist()
        expected_snr = np.full_like(simulated_snr, self.snr).tolist()
        self.assertListEqual(simulated_snr, expected_snr)

    def test_table_columns(self):
        """Test simulated LC table has correct column names"""

        for column in ('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'):
            self.assertIn(column, self.simulated_lc.colnames)

    def test_runs_with_sncosmo(self):
        """Test the Simulated LC can be fit with ``sncosmo``"""

        model = sncosmo.Model(self.source)
        sncosmo.fit_lc(self.simulated_lc, model, vparam_names=['x0', 'x1', 'c'])

    def test_correct_meta_data_values(self):
        """Test simulated LC table has model parameters in meta data"""

        self.assertEqual(self.params, self.simulated_lc.meta)

    def test_sim_values_equal_obs(self):
        """Check if values in the observation table match values in the
        light-curve table.
        """

        for column in ('time', 'band'):
            cadence = self.observations[column]
            sim_vals = self.simulated_lc[column]
            self.assertSequenceEqual(
                list(cadence), list(sim_vals),
                f'Cadence does not match cadence for {column} values')

    def test_default_x0_value(self):
        """Test default x0 is dependent on z"""

        z = 0.5
        expected_x0 = modeling.calc_x0_for_z(z, self.source)
        simulated_lc = modeling.realize_lc(self.obs, self.source, z=z)
        self.assertEqual(expected_x0, simulated_lc.meta['x0'])

    def test_meta_includes_all_params(self):
        """Test all param values are included in meta data, not just those
        specified as kwargs.
        """

        expected_params = modeling.get_model_with_pwv(self.source).param_names
        simulated_lc = modeling.realize_lc(self.obs, self.source, z=0.5)
        meta_params = list(simulated_lc.meta.keys())
        self.assertListEqual(expected_params, meta_params)

    def test_raises_for_z_equals_0(self):
        """Test a value error is raised for simulating z==0"""

        self.assertRaises(ValueError, modeling.realize_lc, self.obs, self.source, z=0)


class IterLCS(TestCase):
    """Tests for the ``iter_lcs`` light-curve iterator"""

    def setUp(self):
        """Create a new light-curve iterator for each test"""

        self.pwv_vals = 0.01, 5
        self.z_vals = 0.01, 1
        self.source = 'salt2-extended'
        self.observations = modeling.create_observations_table()
        self.lc_iter = modeling.iter_lcs(
            self.observations,
            self.source,
            self.pwv_vals,
            self.z_vals,
            verbose=False)

    def test_return_is_iter(self):
        """Test the return is a generator over Tables"""

        self.assertIsInstance(self.lc_iter, types.GeneratorType)
        self.assertIsInstance(next(self.lc_iter), Table)
