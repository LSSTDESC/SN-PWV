"""Tests for the ``snat_sim.pipeline.nodes.LoadPlasticcCadence`` class"""

from unittest import TestCase

from egon.mock import MockTarget

from snat_sim.pipeline.nodes import LoadPlasticcCadence
from snat_sim.plasticc import PLaSTICC


class LoadsPlasticcTable(TestCase):
    """Test the loading of PLaSTICC data into the test_pipeline"""

    @classmethod
    def setUpClass(self) -> None:
        """Use the ``LoadPlasticcCadence`` node to load data into a mock pipeline"""

        # Create a mock pipeline
        cadence = PLaSTICC('alt_sched', 11)
        load_action = LoadPlasticcCadence(cadence, num_processes=0)
        mock_target = MockTarget()

        # Execute the pipeline
        load_action.output.connect(mock_target.input)
        load_action.execute()
        mock_target.execute()

        # pipeline results
        self.packet = mock_target.accumulated_data[0]

        # manually loaded results
        self.snid, self.params, self.cadence = next(cadence.iter_cadence(iter_lim=1, verbose=False))

    def test_snid_params_match_plasticc_iterator(self) -> None:
        self.assertEqual(self.snid, self.packet.snid, 'Simulation parameters do not match.')

    def test_sim_params_match_plasticc_iterator(self) -> None:
        self.assertEqual(self.params, self.packet.sim_params, 'Simulation parameters do not match.')

    def test_cadence_matches_plasticc_iterator(self) -> None:
        self.assertEqual(self.cadence, self.packet.cadence, 'Observational cadences do not match.')
