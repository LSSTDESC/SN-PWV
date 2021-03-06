"""Tests for the ``VariablePWVTrans`` class"""

from unittest import TestCase

from snat_sim import constants as const
from snat_sim import models
from tests.mock import create_constant_pwv_model
from .base import PropagationEffectTests


class BaseTests(PropagationEffectTests, TestCase):
    @classmethod
    def setUpClass(cls):
        mock_pwv_model = create_constant_pwv_model(4)
        cls.propagation_effect = models.VariablePWVTrans(mock_pwv_model)


class DefaultParameterValues(TestCase):
    """Tests for the value of default model parameters"""

    @classmethod
    def setUpClass(cls):
        mock_pwv_model = create_constant_pwv_model(4)
        cls.propagation_effect = models.VariablePWVTrans(mock_pwv_model)

    def test_default_location_params_match_vro(self):
        """Test the default values for the observer location match VRO"""

        self.assertEqual(self.propagation_effect['lat'], const.vro_latitude)
        self.assertEqual(self.propagation_effect['lon'], const.vro_longitude)
        self.assertEqual(self.propagation_effect['alt'], const.vro_altitude)

    def test_default_ra_dec_zero(self):
        """Test the default ra and dec are zero"""

        self.assertEqual(self.propagation_effect['ra'], 0)
        self.assertEqual(self.propagation_effect['dec'], 0)
