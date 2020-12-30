from tempfile import NamedTemporaryFile
from unittest import TestCase

from snat_sim.models import SNModel
from snat_sim.pipeline import FitResultsToDisk, PipelineResult


class WritesHeaderOnSetup(TestCase):

    def setUp(self) -> None:
        self.sn_model = SNModel('salt2')
        self.out_file = NamedTemporaryFile('w+')

        self.write_action = FitResultsToDisk(self.sn_model, self.sn_model, self.out_file.name)
        self.write_action.setup()

    def tearDown(self) -> None:
        self.out_file.close()

    def runTest(self) -> None:
        column_names = PipelineResult.column_names(self.sn_model.param_names, self.sn_model.param_names)
        expected_header = ','.join(column_names) + '\n'
        self.assertEqual(expected_header, self.out_file.readline())
