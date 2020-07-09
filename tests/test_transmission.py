# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``transmission`` module"""

from unittest import TestCase

import numpy as np

from sn_analysis.transmission import trans_for_pwv, PWVTrans


class TestPWVTrans(TestCase):
    """Tests for the addition of PWV to sncosmo models"""

    def setUp(self):
        self.transmission_effect = PWVTrans()

    def test_default_pwv_is_zero(self):
        """Test the default ``pwv`` parameter is 0"""

        self.assertEqual(0, self.transmission_effect['pwv'])

    def test_default_resolution_is_five(self):
        """Test the default ``res`` parameter is 5"""

        self.assertEqual(5, self.transmission_effect['res'])

    def test_propagation_applies_pwv_transmission(self):
        """Test the ``propagate`` applies PWV absorption"""

        # Get the expected transmission
        pwv = res = 5
        wave = np.arange(4000, 5000)
        transmission = trans_for_pwv(pwv=pwv, wave=wave, resolution=res)

        # Get the expected flux
        flux = np.ones_like(wave)
        expected_flux = flux * transmission

        # Get the returned flux
        self.transmission_effect._parameters = [pwv, res]
        propagated_flux = self.transmission_effect.propagate(wave, flux)
        self.assertListEqual(expected_flux.tolist(), propagated_flux.tolist())
