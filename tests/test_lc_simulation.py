"""Tests for the ``simulation`` module"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim.lc_simulation import LCSimulator, ObservedCadence
from snat_sim.models import SNModel


class SetupTasks(TestCase):
    """Generic setup and teardown tasks for tests"""

    def setUp(self) -> None:
        """Define test objects"""

        self.model = SNModel('salt2')
        self.cadence = ObservedCadence(
            obs_times=[-1, 0, 1],
            bands=['sdssr', 'sdssr', 'sdssr'],
            zp=25, zpsys='AB', skynoise=0, gain=1
        )
        self.simulator = LCSimulator(self.model, self.cadence)


class CalcX0ForZ(SetupTasks):
    """Tests for the ``calc_x0_for_z`` function"""

    def test_x0_recovers_absolute_mag(self) -> None:
        """Test returned x0 corresponds to specified magnitude"""

        z = 0.5
        abs_mag = -18
        band = 'standard::b'
        x0 = self.simulator.calc_x0_for_z(z, abs_mag=abs_mag)

        self.model.set(z=z, x0=x0)
        recovered_mag = self.model.source_peakabsmag(band, 'AB')

        # We need to specify a large enough absolute tolerance to account for
        # interpolation error within sncosmo
        np.testing.assert_allclose(abs_mag, recovered_mag, rtol=0, atol=.03)

    def test_model_not_mutated(self) -> None:
        """Test the x0 calculation does not mutate the class model"""

        original_parameters = self.simulator.model.parameters.copy()
        self.simulator.calc_x0_for_z(z=1)
        np.testing.assert_array_equal(self.simulator.model.parameters, original_parameters)


class SimulateFixedSNR(SetupTasks):
    """Tests for the ``simulate_lc_fixed_snr`` function"""

    def setUp(self) -> None:
        """Simulate a light-curve at a fixed SNR"""

        super(SimulateFixedSNR, self).setUp()

        self.snr = 6
        self.params = dict(x1=.8, c=-.5, z=.6, t0=1, x0=1)
        self.simulated_lc = self.simulator.simulate_lc_fixed_snr(snr=self.snr, params=self.params)

    def test_simulated_snr(self) -> None:
        """Test SNR of simulated light-curve equals snr kwarg"""

        simulated_snr = self.simulated_lc['flux'] / self.simulated_lc['fluxerr']
        simulated_snr = np.round(simulated_snr, 3).tolist()
        expected_snr = np.full_like(simulated_snr, self.snr)
        np.testing.assert_array_equal(simulated_snr, expected_snr)

    def test_table_columns(self) -> None:
        """Test simulated LC table has correct column names"""

        for column in ('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'):
            self.assertIn(column, self.simulated_lc.colnames)

    def test_runs_with_sncosmo(self) -> None:
        """Test the Simulated LC can be fit with ``sncosmo``"""

        sncosmo.fit_lc(self.simulated_lc, self.model, vparam_names=['x0', 'x1', 'c'], warn=False)

    def test_correct_meta_data_values(self) -> None:
        """Test simulated LC table has model parameters in meta data"""

        self.assertEqual(self.params, self.simulated_lc.meta)

    def test_default_x0_value(self) -> None:
        """Test default x0 is calculated from the redshift"""

        z = 0.5
        expected_x0 = self.simulator.calc_x0_for_z(z)
        simulated_lc = self.simulator.simulate_lc_fixed_snr(params={'z': z})
        self.assertEqual(expected_x0, simulated_lc.meta['x0'])

    def test_meta_includes_all_params(self) -> None:
        """Test all param values are included in meta data, not just those
        specified as kwargs.
        """

        expected_params = self.model.param_names
        simulated_lc = self.simulator.simulate_lc_fixed_snr(params={'z': 0.5})
        meta_params = list(simulated_lc.meta.keys())
        self.assertListEqual(expected_params, meta_params)

    def test_raises_for_z_equals_0(self) -> None:
        """Test a value error is raised for simulating z == 0"""

        with self.assertRaises(ValueError):
            self.simulator.simulate_lc_fixed_snr(params={'z': 0})


class DuplicatePlasticcSncosmo(TestCase):
    """Tests for the ``duplicate_plasticc_sncosmo`` function"""

    def setUp(self) -> None:
        self.model = models.SNModel('salt2-extended')
        self.plasticc_lc = create_mock_plasticc_light_curve()
        self.param_mapping = {  # Maps sncosmo param names to plasticc names
            't0': 'SIM_PEAKMJD',
            'x1': 'SIM_SALT2x1',
            'c': 'SIM_SALT2c',
            'z': 'SIM_REDSHIFT_CMB',
            'x0': 'SIM_SALT2x0'
        }

    def test_lc_meta_matches_params(self) -> None:
        """Test parameters in returned meta data match the input light_curve"""

        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(self.plasticc_lc, model=self.model, cosmo=None)
        for sncosmo_param, plasticc_param in self.param_mapping.items():
            self.assertEqual(
                duplicated_lc.meta[sncosmo_param], self.plasticc_lc.meta[plasticc_param],
                f'Incorrect {sncosmo_param} ({plasticc_param}) parameter'
            )

    def test_x0_overwritten_by_cosmo_arg(self) -> None:
        """Test the x0 parameter is overwritten according to the given cosmology"""

        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(self.plasticc_lc, model=self.model)
        expected_x0 = calc_x0_for_z(duplicated_lc.meta['z'], source=self.model.source)
        np.testing.assert_allclose(expected_x0, duplicated_lc.meta['x0'])
