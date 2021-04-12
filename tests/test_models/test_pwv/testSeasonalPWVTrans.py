"""Tests for the ``snat_sim.modeling.pwv.SeasonalPWVTrans`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.models import pwv
from . import testVariablePWVTrans
from .base import PropagationEffectTests


class BaseTests(PropagationEffectTests, TestCase):
    """Extends base tests used for all propagation effects"""

    @classmethod
    def setUpClass(cls):
        """Create an instance of the propagation effect for testing

        Set all seasonal PWV values to the same PWV concentration.
        """

        cls.propagation_effect = pwv.SeasonalPWVTrans()
        cls.pwv = 7
        cls.propagation_effect['winter'] = cls.pwv
        cls.propagation_effect['spring'] = cls.pwv
        cls.propagation_effect['summer'] = cls.pwv
        cls.propagation_effect['fall'] = cls.pwv

    def test_propagation_includes_pwv_transmission(self):
        """Test that the ``propagate`` method applies PWV absorption"""

        # Get the expected transmission
        wave = np.arange(4000, 5000)
        transmission_model = pwv.PWVTransmissionModel(resolution=self.propagation_effect._transmission_res)
        transmission = transmission_model.calc_transmission(pwv=self.pwv, wave=wave)

        # Get the returned flux
        propagated_flux = self.propagation_effect.propagate(wave, np.ones_like(wave), time=0)
        np.testing.assert_equal(transmission, propagated_flux[0])


class DefaultParameterValues(testVariablePWVTrans.DefaultParameterValues):
    """Tests for the value of default model parameters"""

    @classmethod
    def setUpClass(cls):
        """Create an instance of the propagation effect for testing"""

        cls.propagation_effect = pwv.SeasonalPWVTrans()

    def test_default_seasonal_values_are_zero(self):
        """Test the default values for the observer location match VRO"""

        self.assertEqual(0, self.propagation_effect['winter'])
        self.assertEqual(0, self.propagation_effect['spring'])
        self.assertEqual(0, self.propagation_effect['summer'])
        self.assertEqual(0, self.propagation_effect['fall'])
