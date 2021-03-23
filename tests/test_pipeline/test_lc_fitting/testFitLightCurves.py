from time import sleep
from typing import List
from unittest import TestCase

import sncosmo
from egon.mock import MockSource, MockTarget

from snat_sim import constants as const
from snat_sim.models import SNModel
from snat_sim.pipeline.nodes.lc_fitting import FitLightCurves


class GenericSetup:
    """Setup tasks shared by multiple test cases"""

    @classmethod
    def generic_setup(cls, vparams: List[str]) -> None:
        """Set up nodes for feeding/accumulating a mock fitting pipeline

        Args:
            vparams: Parameters to vary in the fit
        """

        cls.lc = sncosmo.load_example_data()
        cls.lc.meta['SNID'] = 'id_val'

        source = MockSource([cls.lc])
        cls.node = FitLightCurves(SNModel('salt2-extended'), vparams=vparams, num_processes=0)
        target = MockTarget()

        source.output.connect(cls.node.light_curves_input)
        cls.node.success_output.connect(target.input)
        for mock_node in (source, cls.node, target):
            mock_node.execute()
            sleep(2)

        cls.pipeline_result = target.accumulated_data[0]


class SuccessfulFitOutput(TestCase, GenericSetup):
    """Test node outputs for successful fits match values from the fit"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.generic_setup(['x0'])

    def runTest(self) -> None:
        """Check each individual output value from the fitting node"""

        fitted_result, fitted_model = self.node.fit_lc(self.lc)
        fit_params = dict(zip(fitted_result.param_names, fitted_result.parameters))

        apparent_bmag = fitted_model.source.bandmag('bessellb', 'ab', phase=0),
        absolute_bmag = fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo),

        self.assertEqual(self.lc.meta['SNID'], self.pipeline_result.snid)
        self.assertEqual(self.lc.meta, self.pipeline_result.sim_params)
        self.assertEqual(fit_params, self.pipeline_result.fit_params)
        self.assertEqual(fitted_result.errors, self.pipeline_result.fit_err)
        self.assertEqual(fitted_result.chisq, self.pipeline_result.chisq)
        self.assertEqual(fitted_result.ndof, self.pipeline_result.ndof)
        self.assertEqual(apparent_bmag, self.pipeline_result.mb)
        self.assertEqual(absolute_bmag, self.pipeline_result.abs_mag)
        self.assertEqual('FitLightCurves: ' + fitted_result.message, self.pipeline_result.message)


class FailedFitOutput(TestCase, GenericSetup):
    """Test node outputs for a failed fit are correctly masked"""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up mock nodes for feeding/accumulating a mock pipeline"""

        cls.generic_setup(['z'])

    def runTest(self) -> None:
        """Check each individual output value from the fitting node"""

        self.assertEqual(self.lc.meta['SNID'], self.pipeline_result.snid)
        self.assertEqual(self.lc.meta, self.pipeline_result.sim_params)
        self.assertEqual(dict(), self.pipeline_result.fit_params)
        self.assertEqual(dict(), self.pipeline_result.fit_err)
        self.assertEqual(-99.99, self.pipeline_result.chisq)
        self.assertEqual(-99.99, self.pipeline_result.ndof)
        self.assertEqual(-99.99, self.pipeline_result.mb)
        self.assertEqual(-99.99, self.pipeline_result.abs_mag)
        self.assertEqual('FitLightCurves: z must be bounded if fit.', self.pipeline_result.message)
