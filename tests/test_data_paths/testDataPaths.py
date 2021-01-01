"""Tests for the ``_data_paths`` module"""

import os
from pathlib import Path
from unittest import TestCase

from snat_sim._data_paths import DataPaths


class UsesEnvironmentalVariables(TestCase):
    """Test paths that are configurable via environmental variable use those variables"""

    old_environ: dict

    @classmethod
    def setUpClass(cls) -> None:
        """Store a copy of the original environmental variables"""

        cls.old_environ = os.environ.copy()

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore original environmental variables"""

        del os.environ['SNAT_SIM_DATA']
        del os.environ['CADENCE_SIMS']
        os.environ.update(cls.old_environ)

    def test_root_data_dir(self) -> None:
        """Test the root data directory is determined from the environment at init"""

        test_dir = Path('test_dir').resolve()
        os.environ['SNAT_SIM_DATA'] = str(test_dir)
        self.assertEqual(DataPaths().data_dir, test_dir)

    def test_plasticc_data_dir(self) -> None:
        """Test the plasticc data directory is determined from the environment at init"""

        test_dir = Path('test_dir').resolve()
        os.environ['CADENCE_SIMS'] = str(test_dir)
        self.assertEqual(DataPaths().get_plasticc_dir(), test_dir)


class PLaSTICCPaths(TestCase):
    """Tests the resolution of paths for PLaSTICC data"""

    def setUp(self) -> None:
        """Create a ``DataPaths`` object for testing"""

        self.data_paths = DataPaths()

    def test_error_on_model_without_cadence(self) -> None:
        """Test an error is raised when a model is specified without a cadence"""

        with self.assertRaises(ValueError):
            self.data_paths.get_plasticc_dir(model=11)
