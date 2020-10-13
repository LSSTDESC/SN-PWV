# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``constants`` module"""

from unittest import TestCase

import astropy.units as u

from sn_analysis import constants as const


class Dimensions(TestCase):
    """Test dimensional units have units expected by the rest of the package"""

    def test_vro_latitude_units(self):
        """Test ``constants.vro_latitude`` is in units of degrees"""

        self.assertEqual(const.vro_latitude.unit, u.deg)

    def test_vro_longitude_units(self):
        """Test ``constants.vro_longitude`` is in units of degrees"""

        self.assertEqual(const.vro_longitude.unit, u.deg)

    def test_vro_altitude_units(self):
        """Test ``constants.vro_altitude`` is in units of meters"""

        self.assertEqual(const.vro_altitude.unit, u.m)
