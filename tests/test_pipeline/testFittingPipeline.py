"""Tests for the ``FittingPipeline`` class"""

from unittest import TestCase

from snat_sim.pipeline import FittingPipeline, SNModel


class ValidatePipelineNodes(TestCase):
    """Validate an instance of the fitting test_pipeline"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = FittingPipeline(
            cadence='alt_sched',
            sim_model=SNModel('salt2'),
            fit_model=SNModel('salt2'),
            vparams=['x0'],
            out_path='foo.csv',
        )

    def test_auto_validation(self) -> None:
        """Run the builtin test_pipeline validation routine"""

        self.pipeline.validate()

    def test_limit_on_input_simulations(self) -> None:
        """Test a finite limit is imposed on the number of input PLaSTICC simulations"""

        self.assertLess(self.pipeline.simulate_light_curves.plasticc_data_input.maxsize, float('inf'))


class InitErrors(TestCase):
    """Test for errors raised during pipeline init"""

    def test_error_on_missing_pwv_model(self) -> None:
        """``ValueError`` should be raised if reference stars are specified without a PWV model."""

        with self.assertRaisesRegex(ValueError,
                                    'Cannot perform reference star subtraction without ``pwv_model`` argument'):
            FittingPipeline(
                cadence='alt_sched',
                sim_model=SNModel('salt2'),
                fit_model=SNModel('salt2'),
                vparams=['x0'],
                ref_stars=('G2', 'M5', 'K2'),
                pwv_model=None,
                out_path='foo.csv',
            )
