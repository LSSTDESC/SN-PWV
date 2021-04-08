from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import Table

from tests.mock import create_mock_pipeline_packet


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

        cls.packet = create_mock_pipeline_packet()
        cls.temp_dir = TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up any temporary files"""

        cls.temp_dir.cleanup()

    def test_table_matches_input_lc(self) -> None:
        """Test the data written to disk matches the input data"""

        snid = self.demo_lc.meta['SNID']
        data = read_h5(self.temp_out_file, snid)
        np.testing.assert_array_equal(self.demo_lc, data)
