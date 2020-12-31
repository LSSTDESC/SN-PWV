from unittest import TestCase

import sncosmo
from egon.mock import MockSource, MockTarget

from snat_sim.models import SNModel
from snat_sim.pipeline import FitLightCurves


class ResultObjectValues(TestCase):
    """Test node outputs for successful fits don't have any empty values"""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up mock nodes for feeding/accumulating a ``SimulateLightCurves`` instance"""

        cls.lc = sncosmo.load_example_data()
        cls.lc.meta['SNID'] = 'id_val'

        source = MockSource([cls.lc])
        node = FitLightCurves(SNModel('salt2-extended'), vparams=['x0'], num_processes=0)
        target = MockTarget()

        source.output.connect(node.light_curves_input)
        node.fit_results_output.connect(target.input)
        for mock_node in (source, node, target):
            mock_node.execute()

        cls.pipeline_result = target.accumulated_data[0]
        cls.fitted_result, cls.fitted_model = node.fit_lc(cls.lc)

    def runTest(self) -> None:
        fit_params = dict(zip(self.fitted_result.param_names, self.fitted_result.parameters))

        self.assertEqual(self.lc.meta['SNID'], self.pipeline_result.snid)
        self.assertEqual(self.lc.meta, self.pipeline_result.sim_params)
        self.assertEqual(fit_params, self.pipeline_result.fit_params)
        self.assertEqual(self.fitted_result.errors, self.pipeline_result.fit_err)
        self.assertEqual(self.fitted_result.chisq, self.pipeline_result.chisq)
        self.assertEqual(self.fitted_result.ndof, self.pipeline_result.ndof)
        self.assertEqual(self.fitted_model.mB(), self.pipeline_result.mb)
        self.assertEqual(self.fitted_model.MB(), self.pipeline_result.abs_mag)
        self.assertEqual(self.fitted_result.message, self.pipeline_result.message)
