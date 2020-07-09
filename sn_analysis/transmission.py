# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``transmission`` module handles the calculation of the atmospheric
transmission as a function of wavelength and PWV.
"""

import numpy as np
import sncosmo
from pwv_kpno import pwv_atm
from scipy.stats import binned_statistic


def trans_for_pwv(pwv, wavelengths, resolution):
    """Retrieve the pwv_kpno transmission at given wavelengths and resolution

    Args:
        pwv           (float): The PWV along line of sight
        wavelengths (ndarray): Array of wavelengths to evaluate transmission at
        resolution    (float): Resolution to bin transmission at

    Returns:
        An array of transmission values
    """

    # Create bins that uniformly sample the given wavelength range
    # at the given resolution
    half_res = resolution / 2
    bins = np.arange(
        min(wavelengths) - half_res,
        max(wavelengths) + half_res + resolution,
        resolution)

    # Bin the atm model to the desired resolution
    atm_model = pwv_atm.trans_for_pwv(pwv)
    statistic_left, bin_edges_left, _ = binned_statistic(
        atm_model['wavelength'],
        atm_model['transmission'],
        statistic='mean',
        bins=bins[:-1]
    )

    statistic_right, bin_edges_right, _ = binned_statistic(
        atm_model['wavelength'],
        atm_model['transmission'],
        statistic='mean',
        bins=bins[1:]
    )

    statistic = (statistic_right + statistic_left) / 2

    dx = atm_model['wavelength'][1] - atm_model['wavelength'][0]
    bin_centers = bin_edges_left[:-1] + dx / 2

    # Evaluate the transmission at the desired wavelengths
    return np.interp(wavelengths, bin_centers, statistic)


# Todo: We should bin the PWV transmission to the same resolution as the template
class PWVTrans(sncosmo.PropagationEffect):
    """Atmospheric PWV propagation effect for sncosmo"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self):
        self._param_names = ['pwv', 'res']
        self.param_names_latex = ['PWV', 'resolution']
        self._parameters = np.array([0., 5])

    def propagate(self, wave, flux):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values

        Returns:
            An array of flux values after suffering propagation effects
        """

        pwv, res = self.parameters
        transmission = trans_for_pwv(pwv, wave, res)
        return flux * transmission
