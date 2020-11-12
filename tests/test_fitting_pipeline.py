"""Tests for the ``fitting_pipeline`` module"""

from unittest import TestCase

import sncosmo

from snat_sim.fitting_pipeline import FittingPipeline


class TestProcessAllocation(TestCase):
    """Tests for the allocation of processes to each pipeline task"""

    def test_allocations_sum_to_pool_size(self):
        """Test the number of allocated processes sum to the total pool size"""

        pipeline = FittingPipeline(
            cadence='alt_sched',
            sim_model=sncosmo.Model('salt2'),
            fit_model=sncosmo.Model('salt2'),
            vparams=[],
            pool_size=7
        )

        # include two I/O processes
        allocated_processes = pipeline.simulation_pool_size + pipeline.fitting_pool_size + 2
        self.assertEqual(
            allocated_processes, pipeline.pool_size,
            'Allocated processes do not equal total pool size')

    def test_error_pool_size_less_than_four(self):
        """Test an error is raised if asked to init with less than 4 processes"""

        with self.assertRaises(RuntimeError):
            FittingPipeline(
                cadence='alt_sched',
                sim_model=sncosmo.Model('salt2'),
                fit_model=sncosmo.Model('salt2'),
                vparams=[],
                pool_size=3
            )
