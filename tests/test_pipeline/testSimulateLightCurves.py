"""Tests for the ``SimulateLightCurves`` class"""

from copy import copy
from unittest import TestCase

import numpy as np
from egon.mock import MockSource, MockTarget

from snat_sim.models import SNModel, ObservedCadence
from snat_sim.pipeline import SimulateLightCurves
from tests.mock import create_mock_plasticc_light_curve


class LightCurveSimulation(TestCase):
    """Tests for the ``duplicate_plasticc_sncosmo`` function"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.zp = 35
        cls.plasticc_lc = create_mock_plasticc_light_curve()
        cls.node = SimulateLightCurves(SNModel('salt2-extended'), num_processes=0)
        cls.duplicated_lc = cls.node.duplicate_plasticc_lc(cls.plasticc_lc, zp=cls.zp)

    def test_lc_meta_matches_params(self) -> None:
        """Test parameters in returned meta data match the input light_curve"""

        param_mapping = {  # Maps sncosmo param names to plasticc names
            't0': 'SIM_PEAKMJD',
            'x1': 'SIM_SALT2x1',
            'c': 'SIM_SALT2c',
            'z': 'SIM_REDSHIFT_CMB',
            'ra': 'RA',
            'dec': 'DECL',
            'SNID': 'SNID'
            # The x0 param should be overwritten during the simulation
            # See the ``test_x0_overwritten_by_cosmo_arg`` test
            # 'x0': 'SIM_SALT2x0',
        }

        for sncosmo_param, plasticc_param in param_mapping.items():
            self.assertEqual(
                self.duplicated_lc.meta[sncosmo_param], self.plasticc_lc.meta[plasticc_param],
                f'Incorrect {sncosmo_param} parameter in meta (PLaSTICC parameter {plasticc_param}).'
            )

    def test_x0_overwritten_by_cosmo_arg(self) -> None:
        """Test the x0 parameter is overwritten according to the given cosmology"""

        params, _ = ObservedCadence.from_plasticc(self.plasticc_lc)

        model = copy(self.node.sim_model)
        model.update({p: v for p, v in params.items() if p in model.param_names})
        model.set_source_peakabsmag(self.node.abs_mb, 'standard::b', 'AB', cosmo=self.node.cosmo)
        self.assertEqual(model['x0'], self.duplicated_lc.meta['x0'])

    def test_zp_is_overwritten_with_constant(self) -> None:
        """Test the zero-point of the simulated light_curve is overwritten as a constant"""

        np.testing.assert_equal(self.zp, self.duplicated_lc['zp'])


class ResultRouting(TestCase):
    """Test the routing of pipeline results to the correct nodes"""

    def setUp(self) -> None:
        """Set up mock nodes for feeding/accumulating a ``SimulateLightCurves`` instance"""

        self.source = MockSource()
        self.node = SimulateLightCurves(SNModel('salt2-extended'), num_processes=0)
        self.simulation_target = MockTarget()
        self.failure_target = MockTarget()

        self.source.output.connect(self.node.plasticc_data_input)
        self.node.simulation_output.connect(self.simulation_target.input)
        self.node.failure_result_output.connect(self.failure_target.input)

    def run_nodes(self) -> None:
        """Execute all nodes in the correct order"""

        for node in (self.source, self.node, self.simulation_target, self.failure_target):
            node.execute()

    def test_success_routed_to_simulation_output(self) -> None:
        """Test successful simulations are sent to the ``simulation_output`` connector"""

        plasticc_lc = create_mock_plasticc_light_curve()
        self.source.load_data.append(plasticc_lc)
        self.run_nodes()

        self.assertTrue(self.simulation_target.accumulated_data)
        self.assertFalse(self.failure_target.accumulated_data)

    def test_failure_routed_to_failure_result_output(self) -> None:
        """Test failed simulations are sent to the ``failure_result_output`` connector"""

        plasticc_lc = create_mock_plasticc_light_curve()
        plasticc_lc.meta['SIM_REDSHIFT_CMB'] = 100  # Pick a crazy redshift so the simulation fails
        self.source.load_data.append(plasticc_lc)
        self.run_nodes()

        self.assertFalse(self.simulation_target.accumulated_data)
        self.assertTrue(self.failure_target.accumulated_data)
