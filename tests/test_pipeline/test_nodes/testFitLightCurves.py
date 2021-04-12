"""Tests for the ``snat_sim.pipeline.nodes.FitLightCurves`` class"""

from time import sleep
from typing import List
from unittest import TestCase

import numpy as np
from egon.mock import MockSource, MockTarget

from snat_sim.models import SNModel
from snat_sim.pipeline.nodes import FitLightCurves
from tests.mock import create_mock_pipeline_packet


class GenericSetup:
    """Setup tasks shared by multiple test cases"""

    @classmethod
    def generic_setup(cls, vparams: List[str]) -> None:
        """Set up nodes for feeding/accumulating a mock fitting pipeline

        Args:
            vparams: Parameters to vary in the fit
        """

        cls.packet = create_mock_pipeline_packet(include_fit=False)

        # Create a mock pipeline for fitting the packet's light-curve
        source = MockSource([cls.packet])
        cls.node = FitLightCurves(SNModel('salt2-extended'), vparams=vparams, num_processes=0)
        cls.success_target = MockTarget()
        cls.failure_target = MockTarget()

        source.output.connect(cls.node.light_curves_input)
        cls.node.success_output.connect(cls.success_target.input)
        cls.node.failure_output.connect(cls.failure_target.input)

        # Run the mock pipeline
        for mock_node in (source, cls.node, cls.success_target, cls.failure_target):
            mock_node.execute()
            sleep(2)


class SuccessfulFitOutput(TestCase, GenericSetup):
    """Test node outputs for successful fits match values from the fit"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.generic_setup(['x0'])

    def runTest(self) -> None:
        """Check each individual output value from the fitting node"""

        # Calculate expected values
        fitted_result, fitted_model = self.node.fit_lc(self.packet.light_curve, self.packet.sim_params)

        # Compare those results against actual values
        pipeline_result = self.success_target.accumulated_data[0]
        self.assertEqual(self.packet.snid, pipeline_result.snid)
        self.assertEqual(self.packet.sim_params, pipeline_result.sim_params)
        self.assertEqual(fitted_result, pipeline_result.fit_result)
        np.testing.assert_array_equal(fitted_model.parameters, pipeline_result.fitted_model.parameters)
        self.assertEqual('FitLightCurves: Minimization exited successfully.', pipeline_result.message)


class FailedFitOutput(TestCase, GenericSetup):
    """Test node outputs for a failed fit are correctly masked"""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up mock nodes for feeding/accumulating a mock pipeline"""

        # The fit for this data will fail if we only vary redshift
        cls.generic_setup(['z'])

    def runTest(self) -> None:
        """Check each individual output value from the fitting node"""

        pipeline_result = self.failure_target.accumulated_data[0]
        self.assertEqual(self.packet.snid, pipeline_result.snid)
        self.assertEqual(self.packet.sim_params, pipeline_result.sim_params)
        self.assertIsNone(pipeline_result.fit_result)
        self.assertIsNone(pipeline_result.fitted_model)
        self.assertEqual('FitLightCurves: z must be bounded if fit.', pipeline_result.message)
