"""Tests for the ``FittingPipeline`` class"""

from unittest import TestCase

from snat_sim.pipeline import FittingPipeline, SNModel


class ValidatePipelineNodes(TestCase):
    """Validate an instance of the fitting pipeline"""

    @staticmethod
    def runTest() -> None:
        """Run the builtin pipeline validation routine"""

        FittingPipeline(
            cadence='alt_sched',
            sim_model=SNModel('salt2'),
            fit_model=SNModel('salt2'),
            vparams=['x0'],
            out_path='foo.csv',
        ).validate()
