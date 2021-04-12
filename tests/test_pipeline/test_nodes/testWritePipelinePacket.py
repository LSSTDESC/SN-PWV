"""Tests for the ``snat_sim.pipeline.nodes.WritePipelinePacket`` class"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import h5py
import pandas as pd
from egon.mock import MockSource

from snat_sim.pipeline.nodes import WritePipelinePacket
from tests.mock import create_mock_pipeline_packet


class InputDataMatchesDisk(TestCase):
    """Test the input data is the same as the data written to disk"""

    @classmethod
    def setUpClass(cls) -> None:
        """Use the ``SimulationToDisk`` node to write data to disk"""

        cls.packets = [create_mock_pipeline_packet(123), create_mock_pipeline_packet(456)]
        cls.temp_dir = TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name) / 'tempfile.h5'

        moc_source = MockSource(cls.packets)
        node = WritePipelinePacket(cls.temp_path)
        moc_source.output.connect(node.data_input)

        moc_source.execute()
        node.execute()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up any temporary files"""

        cls.temp_dir.cleanup()

    def test_sim_params_match_packets(self) -> None:
        """Test the simulation parameters written to disk match the input data"""

        data_from_packets = pd.concat([p.sim_params_to_pandas() for p in self.packets]).astype(str)
        data_from_file = pd.read_hdf(self.temp_path, 'simulation/params')
        pd.testing.assert_frame_equal(data_from_packets, data_from_file)

    def test_fit_params_matches_packet(self) -> None:
        """Test the fitted parameters written to disk match the input data"""

        data_from_packets = pd.concat([p.fitted_params_to_pandas() for p in self.packets]).astype(str)
        data_from_file = pd.read_hdf(self.temp_path, 'fitting/params')
        pd.testing.assert_frame_equal(data_from_packets, data_from_file)

    def test_covariance_matches_packet(self) -> None:
        """Test the covariance written to disk matches the input data"""

        packet = self.packets[0]
        covariance = pd.read_hdf(self.temp_path, f'fitting/covariance/{packet.snid}')
        pd.testing.assert_frame_equal(packet.covariance, covariance)

    def test_light_curve_matches_packet(self) -> None:
        """Test the light-curve written to disk matches the input data"""

        packet = self.packets[0]
        light_curve = pd.read_hdf(self.temp_path, f'simulation/lcs/{packet.snid}')
        pd.testing.assert_frame_equal(packet.light_curve.to_pandas(), light_curve)

    def test_all_light_curves_written(self) -> None:
        """Test all light-curves were written to disk"""

        file_obj = h5py.File(self.temp_path)
        available_ids = list(file_obj['simulation/lcs'].keys())
        expected_ids = [str(p.snid) for p in self.packets]
        self.assertListEqual(expected_ids, available_ids)

    def test_all_covariances_written(self) -> None:
        """Test all covariances matrices were written to disk"""

        file_obj = h5py.File(self.temp_path)
        available_ids = list(file_obj['fitting/covariance'].keys())
        expected_ids = [str(p.snid) for p in self.packets]
        self.assertListEqual(expected_ids, available_ids)
