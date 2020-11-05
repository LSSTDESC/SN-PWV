"""Tests for the ``constants`` module"""

from unittest import TestCase

from snat_sim import constants as const


class BetouleCosmology(TestCase):
    """Test the Betoule cosmology object has the same parametervalues as
    those specified the constants package"""

    def runTest(self):
        self.assertEqual(const.betoule_cosmo.H0.value, const.betoule_H0)
        self.assertEqual(const.betoule_cosmo.Om0, const.betoule_omega_m)
