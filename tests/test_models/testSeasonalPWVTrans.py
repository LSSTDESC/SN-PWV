"""Tests for the ``SeasonalPWVTrans`` class"""

from unittest import TestCase

from snat_sim import models
from . import testVariablePWVTrans
from .base import PropagationEffectTests


class BaseTests(PropagationEffectTests, TestCase):

    @classmethod
    def setUpClass(cls):
        cls.propagation_effect = models.SeasonalPWVTrans()
        cls.propagation_effect['winter'] = 3
        cls.propagation_effect['spring'] = 4
        cls.propagation_effect['summer'] = 5
        cls.propagation_effect['fall'] = 6


class DefaultParameterValues(testVariablePWVTrans.DefaultParameterValues):
    """Tests for the value of default model parameters"""

    @classmethod
    def setUpClass(cls):
        cls.propagation_effect = models.SeasonalPWVTrans()

    def test_default_seasonal_values_are_zero(self):
        """Test the default values for the observer location match VRO"""

        self.assertEqual(0, self.propagation_effect['winter'])
        self.assertEqual(0, self.propagation_effect['spring'])
        self.assertEqual(0, self.propagation_effect['summer'])
        self.assertEqual(0, self.propagation_effect['fall'])
