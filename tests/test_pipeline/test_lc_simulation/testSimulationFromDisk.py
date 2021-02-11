from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import sncosmo
from astropy.io.misc.hdf5 import write_table_hdf5
from egon.mock import MockTarget

from snat_sim.pipeline.lc_simultion import SimulationFromDisk


class InputDataMatchesDisk(TestCase):
    """Test that data written to disk matches data loaded by the node"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup a mock pipeline to read dummy data from disk"""

        # Create a demo light-curve table
        cls.demo_lc = sncosmo.load_example_data()
        cls.demo_lc.meta['SNID'] = '12345'

        # Write the light-curve to a temporary file
        cls.temp_out_dir = TemporaryDirectory()
        cls.temp_out_file = Path(cls.temp_out_dir.name) / 'temp_file.h5'
        write_table_hdf5(cls.demo_lc, str(cls.temp_out_file), cls.demo_lc.meta['SNID'])

        # Set up a mock pipeline
        node = SimulationFromDisk(cls.temp_out_file, num_processes=0)
        cls.accumulator = MockTarget()
        node.simulation_output.connect(cls.accumulator.input)

        # Write the data to disk
        node.execute()
        cls.accumulator.execute()

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary files"""

        cls.temp_out_dir.cleanup()

    def runTest(self) -> None:
        np.testing.assert_array_equal(self.demo_lc, self.accumulator.accumulated_data[0])
