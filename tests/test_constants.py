"""Tests for the ``constants`` module"""

from unittest import TestCase

from astropy.cosmology import FlatLambdaCDM

from snat_sim import constants as const


class BetouleCosmology(TestCase):
    """Tests for the ``constants.betoule_cosmo`` object"""

    def test_cosmology_parameters(self) -> None:
        """Test the Betoule cosmology object has the same parameter values as
        those specified the constants package"""

        self.assertEqual(const.betoule_cosmo.H0.value, const.betoule_H0)
        self.assertEqual(const.betoule_cosmo.Om0, const.betoule_omega_m)

    def test_cosmology_is_flat(self) -> None:
        """Test the cosmology object is an instance of ``FlatLambdaCDM``"""

        self.assertIsInstance(const.betoule_cosmo, FlatLambdaCDM)
