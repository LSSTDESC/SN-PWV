"""Tests for the ``ProcessManager`` class"""

from multiprocessing import Process
from time import sleep
from unittest import TestCase

from snat_sim.fitting_pipeline import ProcessManager


def dummy_func():
    """Sleep for ten seconds"""

    sleep(10)


class StartStopCommands(TestCase):
    """Test processes are launched and terminated on command"""

    def setUp(self):
        """Create a ``ProcessManager`` instance with a single dummy process"""

        self.process = Process(target=dummy_func)
        self.manager = ProcessManager([self.process])

    def runTest(self):
        """Launch and then kill the process manager"""

        self.assertFalse(self.process.is_alive())

        self.manager.run_async()
        sleep(1)  # Give the process time to start
        self.assertTrue(self.process.is_alive())

        self.manager.kill()
        sleep(1)  # Give the process time to exit
        self.assertFalse(self.process.is_alive())
