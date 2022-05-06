"""Tests for the ``snat_sim.pipeline.nodes.LoadPlasticcCadence`` class"""

from unittest import TestCase

import numpy as np
from egon.mock import MockTarget

from snat_sim.pipeline.nodes import LoadPlasticcCadence
from snat_sim.plasticc import PLAsTICC


class LoadsCorrectSimulationData(TestCase):
    """Test the loading of PLAsTICC data into a test pipeline"""

    @classmethod
    def setUpClass(cls) -> None:
        """Use the ``LoadPlasticcCadence`` node to load data into a mock pipeline"""

        # Create a mock pipeline
        cadence = PLAsTICC('alt_sched', 11)
        load_action = LoadPlasticcCadence(cadence)
        mock_target = MockTarget()
        load_action.output.connect(mock_target.input)

        # Execute the pipeline
        load_action.execute()
        mock_target.execute()

        # pipeline results
        cls.packet = mock_target.accumulated_data[0]

        # manually loaded results
        cls.snid, cls.params, cls.cadence = next(cadence.iter_cadence(iter_lim=1, verbose=False))

    def test_snid_matches_plasticc_iterator(self) -> None:
        """Test the SNID matches the value from the PLAsTICC iterator"""

        self.assertEqual(self.snid, self.packet.snid, 'SNID does not match.')

    def test_sim_params_match_plasticc_iterator(self) -> None:
        """Test the simulation parameters match the PLAsTICC iterator"""

        self.assertEqual(self.params, self.packet.sim_params, 'Simulation parameters do not match.')

    def test_cadence_matches_plasticc_iterator(self) -> None:
        """Test the cadence data matches the PLAsTICC iterator"""

        self.assertEqual(self.cadence, self.packet.cadence, 'Observational cadence does not match.')

    def test_zero_point_is_overwritten(self) -> None:
        """Test the zero point of the loaded cadence is overwritten"""

        expected_zp = 30
        np.testing.assert_array_equal(self.packet.cadence.zp, expected_zp)


class NumProcessesLimitedToOne(TestCase):
    """Test the number of allocated processes is limited to one"""

    def runTest(self) -> None:
        cadence = PLAsTICC('alt_sched', 11)
        with self.assertRaises(RuntimeError):
            LoadPlasticcCadence(cadence, num_processes=2)
