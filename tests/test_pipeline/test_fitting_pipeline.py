"""Tests for the ``FittingPipeline`` class"""

from unittest import TestCase

from snat_sim.pipeline import FittingPipeline, SNModel


class ValidatePipelineNodes(TestCase):
    """Validate an instance of the fitting test_pipeline"""

    @staticmethod
    def runTest() -> None:
        """Run the builtin test_pipeline validation routine"""

        FittingPipeline(
            cadence='alt_sched',
            sim_model=SNModel('salt2'),
            fit_model=SNModel('salt2'),
            vparams=['x0'],
            out_path='foo.csv',
        ).validate()


class InitErrors(TestCase):
    """Test for errors raised during pipeline init"""

    def test_error_on_missing_pwv_model(self) -> None:
        """``ValueError`` should be raised if reference stars are specified without a PWV model."""

        with self.assertRaisesRegex(ValueError, 'Cannot perform reference star subtraction without ``pwv_model`` argument'):
            FittingPipeline(
                cadence='alt_sched',
                sim_model=SNModel('salt2'),
                fit_model=SNModel('salt2'),
                vparams=['x0'],
                ref_stars=('G2', 'M5', 'K2'),
                pwv_model=None,
                out_path='foo.csv',
            )
