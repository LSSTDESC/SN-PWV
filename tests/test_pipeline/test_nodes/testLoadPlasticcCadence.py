"""Tests for the ``LoadPlasticcSims`` class"""

from unittest import TestCase

from egon.mock import MockTarget

from snat_sim.models import ObservedCadence
from snat_sim.pipeline.nodes import LoadPlasticcCadence
from snat_sim.plasticc import PLaSTICC


class LoadsPlasticcTable(TestCase):
    """Test the loading of PLaSTICC data into the test_pipeline"""

    def setUp(self) -> None:
        """Use the ``LoadPlasticcCadence`` node to load data into a mock pipeline"""

        self.cadence = PLaSTICC('alt_sched', 11)
        self.load_action = LoadPlasticcCadence(self.cadence, num_processes=0)
        self.mock_target = MockTarget()

        self.load_action.output.connect(self.mock_target.input)
        self.load_action.execute()
        self.mock_target.execute()

    def test_data_matches_plastic_sims(self) -> None:
        """Test that the first loaded data table matches data from the cadence specified at init"""

        # Manually load the data we expect to be loaded by the node
        # noinspection PyTypeChecker
        plasticc_light_curve = next(self.cadence.iter_lc(iter_lim=1, verbose=False))
        params, cadence = ObservedCadence.from_plasticc(plasticc_light_curve)

        packet = self.mock_target.accumulated_data[0]
        self.assertEqual(params, packet.sim_params, 'Simulation parameters do not match.')
        self.assertEqual(cadence, packet.cadence, 'Observational cadences do not match.')
