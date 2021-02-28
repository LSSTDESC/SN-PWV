from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import h5py
import numpy as np
import sncosmo
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import Table
from egon.mock import MockSource

from snat_sim.pipeline.lc_simulation import SimulationToDisk


def read_h5(*args, **kwargs) -> Table:
    """Wrapper for astropy ``read_table_hdf5`` function that automatically decodes bytes into strings"""

    data = read_table_hdf5(*args, **kwargs)
    cols_to_cast = [c for c, cdata in data.columns.items() if cdata.dtype.type == np.bytes_]
    for column in cols_to_cast:
        data[column] = data[column].astype(str)

    return data


class InputDataMatchesDisk(TestCase):
    """Test the input data is the same as the data written to disk"""

    @classmethod
    def setUpClass(cls) -> None:
        """Use the ``SimulationToDisk`` node to write data to disk"""

        # Create a demo light-curve table
        cls.demo_lc = sncosmo.load_example_data()
        cls.demo_lc.meta['SNID'] = '12345'
        source = MockSource([cls.demo_lc])

        # Set up the node to write to a temp file
        cls.temp_out_dir = TemporaryDirectory()

        cls.temp_out_file = Path(cls.temp_out_dir.name) / 'temp_file.h5'
        node = SimulationToDisk(cls.temp_out_file, num_processes=0)
        source.output.connect(node.simulation_input)

        # Write the data to disk
        source.execute()
        node.execute()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up any temporary files"""

        cls.temp_out_dir.cleanup()

    def test_path_matches_snid(self) -> None:
        """Test the internal HDF5 path matches the light-curve's SNID value"""

        snid = self.demo_lc.meta['SNID']
        with h5py.File(self.temp_out_file) as h5_data:
            self.assertListEqual([snid], list(h5_data.keys()))

    def test_table_matches_input_lc(self) -> None:
        """Test the data written to disk matches the input data"""

        snid = self.demo_lc.meta['SNID']
        data = read_h5(self.temp_out_file, snid)
        np.testing.assert_array_equal(self.demo_lc, data)
