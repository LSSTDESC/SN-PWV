# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Defines values for constants used across the parent package."""

from astropy import units as _u
from astropy.cosmology import FlatLambdaCDM as _FlatLambdaCDM

# Vera Rubin Observatory coordinates from Google Maps
vro_latitude = -30.244573 * _u.deg
vro_longitude = -70.7499537 * _u.deg
vro_altitude = 1024 * _u.m

# Fitted parameters from the cosmological analysis of Betoule et al. 2014
betoule_alpha = 0.141
betoule_beta = 3.101
betoule_omega_m = 0.295
betoule_abs_mb = -19.05
betoule_H0 = 70
betoule_cosmo = _FlatLambdaCDM(H0=betoule_H0, Om0=betoule_omega_m)
