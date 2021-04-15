"""Tests for the ``snat_sim.pipeline.nodes.LightCurveSimulation`` class"""

from unittest import TestCase

import pandas as pd
from egon.mock import MockSource, MockTarget

from snat_sim.models import SNModel
from snat_sim.pipeline.nodes import SimulateLightCurves
from tests.mock import create_mock_pipeline_packet, create_mock_variable_catalog


class ResultRouting(TestCase):
    """Test the routing of pipeline results to the correct nodes"""

    def setUp(self) -> None:
        """Set up mock nodes for feeding/accumulating a ``SimulateLightCurves`` instance"""

        # Set up separate target node for each of the ``SimulateLightCurves`` output connectors
        self.source = MockSource()
        self.node = SimulateLightCurves(SNModel('salt2-extended'), num_processes=0)
        self.success_target = MockTarget()
        self.failure_target = MockTarget()

        self.source.output.connect(self.node.input)
        self.node.success_output.connect(self.success_target.input)
        self.node.failure_output.connect(self.failure_target.input)

    def run_nodes(self) -> None:
        """Execute all nodes in the correct order"""

        for node in (self.source, self.node, self.success_target, self.failure_target):
            node.execute()

    def test_simulation_routed_to_success_output(self) -> None:
        """Test successful simulations are sent to the ``success_output`` connector"""

        self.source.load_data.append(create_mock_pipeline_packet(include_lc=False))
        self.run_nodes()

        self.assertTrue(self.success_target.accumulated_data)
        self.assertFalse(self.failure_target.accumulated_data)

    def test_failure_routed_to_failure_output(self) -> None:
        """Test failed simulations are sent to the ``failure_output`` connector"""

        # Pick a crazy redshift so the simulation fails
        packet = create_mock_pipeline_packet(include_lc=False)
        packet.sim_params['z'] = 1000

        self.source.load_data.append(packet)
        self.run_nodes()

        self.assertFalse(self.success_target.accumulated_data)
        self.assertTrue(self.failure_target.accumulated_data)


class LightCurveSimulation(TestCase):
    """Tests relating to the simulation of light-curves by the node"""

    @staticmethod
    def test_simulation_includes_reference_catalog() -> None:
        """Test simulated light-curves are calibrated if a reference catalog is specified"""

        model = SNModel('salt2-extended')
        catalog = create_mock_variable_catalog('G2', 'M5', 'K2')
        packet = create_mock_pipeline_packet(include_lc=False)

        print(model.parameters)
        node_without_catalog = SimulateLightCurves(model, add_scatter=False)
        uncalibrated_lc = node_without_catalog.simulate_lc(packet.sim_params, packet.cadence)

        print(model.parameters)
        node_with_catalog = SimulateLightCurves(model, add_scatter=False, catalog=catalog)
        calibrated_lc = node_with_catalog.simulate_lc(packet.sim_params, packet.cadence)

        ra, dec = packet.sim_params['ra'], packet.sim_params['dec']
        pd.testing.assert_frame_equal(
            catalog.calibrate_lc(uncalibrated_lc, ra, dec).to_pandas(),
            calibrated_lc.to_pandas()
        )
