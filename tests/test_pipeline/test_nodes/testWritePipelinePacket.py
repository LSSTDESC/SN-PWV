"""Tests for the ``snat_sim.pipeline.nodes.WritePipelinePacket`` class"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
from egon.mock import MockSource

from snat_sim.pipeline.nodes import WritePipelinePacket
from tests.mock import create_mock_pipeline_packet


class InputDataMatchesDisk(TestCase):
    """Test the input data is the same as the data written to disk"""

    @classmethod
    def setUpClass(cls) -> None:
        """Use the ``SimulationToDisk`` node to write data to disk"""

        cls.packet = create_mock_pipeline_packet()
        cls.temp_dir = TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name) / 'tempfile.h5'

        moc_source = MockSource([cls.packet])
        node = WritePipelinePacket(cls.temp_path)
        moc_source.output.connect(node.data_input)

        moc_source.execute()
        node.execute()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up any temporary files"""

        cls.temp_dir.cleanup()

    def test_table_data_matches_input_packet(self) -> None:
        """Test the data written to disk matches the input data"""

        sim_params = pd.read_hdf(self.temp_path, 'simulation/params').iloc[0]
        self.assertEqual(self.packet.sim_params, dict(sim_params))

        light_curve = pd.read_hdf(self.temp_path, f'simulation/lcs/{self.packet.snid}')
        pd.testing.assert_frame_equal(self.packet.light_curve, light_curve)

        covariance = pd.read_hdf(self.temp_path, f'fitting/covariance/{self.packet.snid}')
        pd.testing.assert_frame_equal(self.packet.covariance, covariance)

        fit_params = pd.read_hdf(self.temp_path, 'fitting/params').iloc[0]
        self.assertEqual(self.packet.fitted_params_to_pandas(), dict(fit_params))
