"""Tests for the ``FittingPipeline`` class"""

from tempfile import NamedTemporaryFile
from unittest import TestCase

import sncosmo

from snat_sim.fitting_pipeline import FittingPipeline


class ProcessAllocation(TestCase):
    """Tests for the allocation of processes to each pipeline task"""

    def setUp(self):
        self.temp_file = NamedTemporaryFile()

    def tearDown(self):
        self.temp_file.close()

    def test_allocations_sum_to_pool_size(self):
        """Test the number of allocated processes sum to the total pool size"""

        pool_size = 7
        pipeline = FittingPipeline(
            cadence='alt_sched',
            sim_model=sncosmo.Model('salt2'),
            fit_model=sncosmo.Model('salt2'),
            vparams=[],
            pool_size=pool_size,
            out_path='test.csv'
        )

        self.assertEqual(
            len(pipeline._processes), pool_size,
            'Allocated processes do not equal pool size specified at init')

    def test_error_pool_size_less_than_four(self):
        """Test an error is raised if asked to init with less than 4 processes"""

        with self.assertRaises(RuntimeError):
            FittingPipeline(
                cadence='alt_sched',
                sim_model=sncosmo.Model('salt2'),
                fit_model=sncosmo.Model('salt2'),
                vparams=[],
                pool_size=3,
                out_path='test.csv'
            )
