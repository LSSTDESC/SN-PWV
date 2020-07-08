# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``transmission`` module"""

from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from pwv_kpno import pwv_atm

from sn_analysis import modeling


class TestTransmissionEffects(TestCase):
    """Tests for the addition of PWV to sncosmo models"""

    def setUp(self):
        self.pwv = 10
        self.z = .8
        self.model = modeling.get_model_with_pwv('salt2-extended')

    def test_recovered_transmission(self):
        """The simulated SN flux with PWV / the flux without PWV should be
        equivalent to the PWV transmission function
        """

        wavelengths = np.arange(4000, 10000)

        self.model.set(pwv=0, z=self.z)
        flux = self.model.flux(0, wavelengths)

        self.model.set(pwv=self.pwv, z=self.z)
        flux_pwv = self.model.flux(0, wavelengths)

        transmission = pwv_atm.trans_for_pwv(self.pwv)
        interp_transmission = np.interp(
            wavelengths,
            transmission['wavelength'],
            transmission['transmission'])

        is_close = np.isclose(interp_transmission, flux_pwv / flux).all()
        if not is_close:
            plt.plot(wavelengths, flux_pwv / flux)
            plt.title('Incorrect Recovered Transmission')
            plt.xlabel('Wavelength')
            plt.ylabel('(Flux PWV=10) / (Flux PWV=0)')
            plt.show()

        self.assertTrue(is_close)
