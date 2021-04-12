"""Tests for the ``snat_sim.pipeline.nodes.LightCurveSimulation`` class"""

from unittest import TestCase

import numpy as np
from egon.mock import MockSource, MockTarget

from snat_sim.models import ObservedCadence, SNModel
from snat_sim.pipeline.data_model import PipelinePacket
from snat_sim.pipeline.nodes import SimulateLightCurves
from tests.mock import create_mock_plasticc_light_curve


class LightCurveSimulation(TestCase):
    """Tests for the ``duplicate_plasticc_sncosmo`` function"""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up mock nodes for feeding/accumulating a ``SimulateLightCurves`` instance"""

        cls.plasticc_lc = create_mock_plasticc_light_curve()
        cls.test_node = SimulateLightCurves(SNModel('salt2-extended'), num_processes=0)

        cls.sim_params, cls.sim_cadence = ObservedCadence.from_plasticc(cls.plasticc_lc)
        cls.duplicated_lc = cls.test_node.duplicate_plasticc_lc(cls.sim_params, cls.sim_cadence)

    def test_zp_is_overwritten_with_constant(self) -> None:
        """Test the zero-point of the simulated light_curve is overwritten as a constant"""

        expected_zp = 30
        np.testing.assert_equal(expected_zp, self.duplicated_lc['zp'].values)


class ResultRouting(TestCase):
    """Test the routing of pipeline results to the correct nodes"""

    def setUp(self) -> None:
        """Set up mock nodes for feeding/accumulating a ``SimulateLightCurves`` instance"""

        # Set up separate target node for each of the ``SimulateLightCurves`` output connectors
        self.source = MockSource()
        self.node = SimulateLightCurves(SNModel('salt2-extended'), num_processes=0)
        self.success_target = MockTarget()
        self.failure_target = MockTarget()

        self.source.output.connect(self.node.cadence_data_input)
        self.node.success_output.connect(self.success_target.input)
        self.node.failure_output.connect(self.failure_target.input)

    def run_nodes(self) -> None:
        """Execute all nodes in the correct order"""

        for node in (self.source, self.node, self.success_target, self.failure_target):
            node.execute()

    def test_success_routed_to_simulation_output(self) -> None:
        """Test successful simulations are sent to the ``simulation_output`` connector"""

        params, cadence = ObservedCadence.from_plasticc(create_mock_plasticc_light_curve())
        packet = PipelinePacket(123456, sim_params=params, cadence=cadence)
        self.source.load_data.append(packet)
        self.run_nodes()

        self.assertTrue(self.success_target.accumulated_data)
        self.assertFalse(self.failure_target.accumulated_data)

    def test_failure_routed_to_failure_result_output(self) -> None:
        """Test failed simulations are sent to the ``failure_result_output`` connector"""

        params, cadence = ObservedCadence.from_plasticc(create_mock_plasticc_light_curve())
        params['z'] = 1000  # Pick a crazy redshift so the simulation fails
        packet = PipelinePacket(123456, sim_params=params, cadence=cadence)
        self.source.load_data.append(packet)
        self.run_nodes()

        self.assertFalse(self.success_target.accumulated_data)
        self.assertTrue(self.failure_target.accumulated_data)
