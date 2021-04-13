"""Tests for the ``snat_sim.pipeline.pipelines.FittingPipeline`` class"""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import TestCase

from snat_sim.models import SNModel
from snat_sim.pipeline import FittingPipeline


class ValidatePipeline(TestCase):
    """Validate an instance of the fitting pipeline"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = TemporaryDirectory()
        cls.pipeline = FittingPipeline(
            cadence='alt_sched',
            sim_model=SNModel('salt2'),
            fit_model=SNModel('salt2'),
            vparams=['x0'],
            out_path=Path(cls.temp_dir.name) / 'foo.h5'
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_auto_validation(self) -> None:
        """Run the builtin pipeline validation routine"""

        self.pipeline.validate()

    def test_limit_on_input_simulations(self) -> None:
        """Test a finite limit is imposed on the number of input PLaSTICC simulations"""

        self.assertLess(self.pipeline.simulate_light_curves.input.maxsize, float('inf'))


class OverwriteProtection(TestCase):
    """Test the class will not overwrite existing files by default"""

    def runTest(self):
        with NamedTemporaryFile() as temp_file:
            path = Path(temp_file.name)
            path.touch()
            with self.assertRaises(FileExistsError):
                FittingPipeline(
                    cadence='alt_sched',
                    sim_model=SNModel('salt2'),
                    fit_model=SNModel('salt2'),
                    vparams=['x0'],
                    out_path=path
                )