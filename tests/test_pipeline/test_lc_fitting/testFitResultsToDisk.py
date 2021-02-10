"""Tests for the ``FitResultsToDisk`` class"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from snat_sim.pipeline.lc_fitting import FitResultsToDisk


class FileFormatting(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = TemporaryDirectory()
        out_path = Path(cls.temp_dir.name) / 'out_data.csv'

        node = FitResultsToDisk()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_file_header(self):
        """Test the first line in the file is a header with column names"""

        self.fail()

    def test_csv_format(self):
        """Test data is written in CSV format"""

        self.fail()

    def test_all_inputs_to_disk(self):
        """Test all input values are written to disk on separate lines"""

        self.fail()
