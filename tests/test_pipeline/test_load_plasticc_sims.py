from unittest import TestCase

import numpy as np
from egon.mock import MockTarget

from snat_sim import plasticc
from snat_sim.pipeline import LoadPlasticcSims


class LoadsPlasticcTable(TestCase):
    """Test the loading of PLaSTICC data into the test_pipeline"""

    def setUp(self) -> None:
        self.cadence = 'alt_sched'
        self.load_action = LoadPlasticcSims(self.cadence, iter_lim=1, num_processes=0)
        self.mock_target = MockTarget()

        self.load_action.lc_output.connect(self.mock_target.input)
        self.load_action.execute()
        self.mock_target.execute()

    def test_data_matches_plastic_sims(self) -> None:
        """Test that the first loaded data table matches data from the cadence specified at init"""

        plasticc_data = next(plasticc.iter_lc_for_cadence_model(self.cadence, model=11, verbose=False))
        loaded_data = self.mock_target.accumulated_data[0]
        np.testing.assert_array_equal(plasticc_data, loaded_data)
