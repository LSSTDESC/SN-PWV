"""Tests for the ``FitResultsToDisk`` class"""

from tempfile import NamedTemporaryFile
from unittest import TestCase

from egon.mock import MockSource

from snat_sim.models import SNModel
from snat_sim.pipeline import FitResultsToDisk, PipelineResult


class WritesHeaderOnSetup(TestCase):
    """Test the node writes a CSV header at the top of the file"""

    def setUp(self) -> None:
        """Create a temporary file the node to write to"""

        self.out_file = NamedTemporaryFile('w+')

    def tearDown(self) -> None:
        """Close the temporary file"""

        self.out_file.close()

    def runTest(self) -> None:
        # Let the node write the header to the output fill
        sn_model = SNModel('salt2')
        node = FitResultsToDisk(sn_model, sn_model, self.out_file.name, 0)
        node.setup()

        # Check the output file for a csv header
        model_parameters = sn_model.param_names
        column_names = PipelineResult.column_names(model_parameters, model_parameters)
        expected_header = ','.join(column_names) + '\n'
        self.assertEqual(expected_header, self.out_file.readline())


class ResultsWrittenToFile(TestCase):
    """Test result objects are written to file"""

    def setUp(self) -> None:
        """Create a pipeline node"""

        self.sn_model = SNModel('salt2')
        self.out_file = NamedTemporaryFile('w+')
        self.result = PipelineResult('snid_val')

        source = MockSource([self.result], 0)
        self.node = FitResultsToDisk(self.sn_model, self.sn_model, self.out_file.name, 0)
        source.output.connect(self.node.fit_results_input)

        source.execute()
        self.node.execute()

    def tearDown(self) -> None:
        """Close the temporary file"""

        self.out_file.close()

    def runTest(self) -> None:
        last_line = self.out_file.readlines()[-1]
        expected_line = self.result.to_csv(self.sn_model.param_names, self.sn_model.param_names)
        self.assertEqual(expected_line, last_line)
