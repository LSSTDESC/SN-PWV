"""Tests for the ``snat_sim.modeling.pwv.StaticPWVTrans`` class"""

from unittest import TestCase

import numpy as np

from snat_sim.models import pwv
from .base import PropagationEffectTests


class BaseTests(PropagationEffectTests, TestCase):
    """Extends base tests used for all propagation effects"""

    @classmethod
    def setUpClass(cls):
        """Create an instance of the propagation effect for testing"""

        cls.propagation_effect = pwv.StaticPWVTrans()

    def test_propagation_includes_pwv_transmission(self):
        """Test the ``propagate`` method applies PWV absorption"""

        # Get the expected transmission
        pwv_concentration = 5
        wave = np.arange(4000, 5000)
        transmission_model = pwv.PWVTransmissionModel(resolution=self.propagation_effect._transmission_res)
        transmission = transmission_model.calc_transmission(pwv=pwv_concentration, wave=wave)

        # Get the returned flux
        self.propagation_effect._parameters = [pwv_concentration]
        propagated_flux = self.propagation_effect.propagate(wave, np.ones_like(wave))
        np.testing.assert_equal(transmission, propagated_flux[0])


class DefaultParameterValues(TestCase):
    """Tests for the value of default model parameters"""

    @classmethod
    def setUpClass(cls):
        cls.propagation_effect = pwv.StaticPWVTrans()

    def test_default_pwv_is_zero(self):
        """Test the default ``pwv`` parameter is 0"""

        self.assertEqual(0, self.propagation_effect['pwv'])
