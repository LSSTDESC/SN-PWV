from tempfile import NamedTemporaryFile
from unittest import TestCase

import numpy as np
from egon.mock import MockTarget

from snat_sim.pipeline import LoadPlasticcSims, plasticc, FitResultsToDisk, DataModel, SNModel


class TestLoadPlasticcSims(TestCase):
    """Test the loading of PLaSTICC data into the pipeline"""

    def setUp(self) -> None:
        self.cadence = 'alt_sched'
        self.load_action = LoadPlasticcSims(self.cadence, iter_lim=1, num_processes=0)
        self.mock_target = MockTarget()

        self.load_action.lc_output.connect(self.mock_target.input)
        self.load_action.execute()

    def test_data_matches_plastic_sims(self) -> None:
        """Test that the first loaded data table matches data from the cadence specified at init"""

        plasticc_data = next(plasticc.iter_lc_for_cadence_model(self.cadence, model=11))
        loaded_data = next(self.mock_target.input.iter_get())
        np.testing.assert_array_equal(plasticc_data, loaded_data)


class TestFitResultsToDisk(TestCase):

    def setUp(self) -> None:
        sn_model = SNModel('salt2')
        self.out_file = NamedTemporaryFile()
        self.data_model = DataModel(sn_model, sn_model)
        self.write_action = FitResultsToDisk(self.data_model, self.out_file.name)

    def tearDown(self) -> None:
        self.out_file.close()
