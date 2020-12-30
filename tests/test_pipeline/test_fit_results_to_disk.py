from tempfile import NamedTemporaryFile
from unittest import TestCase

from egon.mock import MockSource

from snat_sim.models import SNModel
from snat_sim.pipeline import FitResultsToDisk, PipelineResult


class TestFitResultsToDisk(TestCase):

    def setUp(self) -> None:
        self.sn_model = SNModel('salt2')
        self.out_file = NamedTemporaryFile('w+')

        self.write_action = FitResultsToDisk(self.sn_model, self.sn_model, self.out_file.name)
        MockSource().output.connect(self.write_action.fit_results_input)

        self.write_action.execute()

    def tearDown(self) -> None:
        self.out_file.close()

    def test_header_is_written(self) -> None:
        expected_header = ','.join(PipelineResult.column_names(self.sn_model.param_names, self.sn_model.param_names)) + '\n'
        self.assertEqual(expected_header, self.out_file.readline())
