# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module handles the simulation of SN light-curves."""

import itertools
from pathlib import Path

import numpy as np
import sncosmo
from astropy.cosmology import WMAP9
from astropy.table import Column, Table
from pwv_kpno import pwv_atm
from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'


###############################################################################
# For building sncosmo models with a PWV component
###############################################################################


# Todo: We should bin the PWV transmission to the same resolution as the template
class PWVTrans(sncosmo.PropagationEffect):
    """Atmospheric PWV propagation effect for sncosmo"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self):
        self._param_names = ['pwv']
        self.param_names_latex = ['PWV']
        self._parameters = np.array([0.])

    def propagate(self, wave, flux):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values

        Returns:
            An array of flux values after suffering propagation effects
        """

        pwv = self.parameters[0]
        transmission = pwv_atm.trans_for_pwv(pwv)
        interp_transmission = np.interp(
            wave, transmission['wavelength'], transmission['transmission'])

        return interp_transmission * flux


# Todo: Extend this to include other atm models as a kwarg
def get_model_with_pwv(source, **params):
    """Return an sncosmo model with PWV effects

    The returned model has the additional parameters:
        pwv: The zenith PWV density in mm

    Args:
        source: An sncosmo source
        Any parameter values to set in the model

    Returns:
        An sncosmo model
    """

    params.setdefault('pwv', 0)
    model = sncosmo.Model(source)
    model.add_effect(PWVTrans(), '', 'obs')
    model.update(params)
    return model


###############################################################################
# For simulating light-curves
###############################################################################


def calc_x0_for_z(z, cosmo=WMAP9, source='salt2-extended'):
    """Set the redshift and corresponding x_0 value of an sncosmo model

    x_0 is determined for the given redshift using the given cosmology.
    A derivation is summarized as follows.

    The apparent magnitude as a function of x_0 is given by:
        m = -2.5 * log(f / f_0)
          = -2.5 * log(x_0 * f|_{x_0=1} / f_0)
          = -2.5 * log(f|_{x_0=1} / f_0) - 2.5 * log(x_0)
          = m|_{x_0=1} - 2.5 * log(x_0)

    The distance module can thus be written as
        mu = M - m = M - m|_{x_0=1} + 2.5 * log(x_0)

    which gives us x_0 as:
       x_0 = 10 ** ((mu - M + m|_{x_0=1}) / 2.5)

    Args:
         z         (float): Model redshift to set
         cosmo (Cosmology): Cosmology to use when determining x0
         source   (Source): Model source to use when determining x0
    """

    if z == 0:
        return 1

    abs_mag = -19.1
    model = sncosmo.Model(source)
    model.set(z=z, x0=1)
    model.set_source_peakabsmag(abs_mag, 'standard::b', 'AB', cosmo=cosmo)

    apparent_mag = model.bandmag('standard::b', 'AB', 0)
    dist_mod_model = apparent_mag - abs_mag
    dist_mod_cosmo = cosmo.distmod(z).value
    return 10 ** ((dist_mod_cosmo - dist_mod_model) / 2.5)


def create_observations_table(
        phases=range(-20, 50),
        bands=('decam_g', 'decam_r', 'decam_i', 'decam_z', 'decam_y'),
        zp=25,
        zpsys='ab'):
    """Create an astropy table defining a uniform observation cadence for a single target

    Time values are specified in units of phase

    Args:
        phases (ndarray): Array of phase values to include
        bands  (ndarray): Array of bands to include
        zp       (float): The zero point
        zpsys    (float): The zero point system

    Returns:
        An astropy table
    """

    phase_arr = np.concatenate([phases for _ in bands])
    band_arr = np.concatenate([bands for _ in phases])
    gain_arr = np.ones_like(phase_arr)
    skynoise_arr = np.zeros_like(phase_arr)
    zp_arr = np.full_like(phase_arr, zp)
    zp_sys_arr = np.full_like(phase_arr, zpsys, dtype='U10')

    observations = Table(
        {'time': phase_arr,
         'band': band_arr,
         'gain': gain_arr,
         'skynoise': skynoise_arr,
         'zp': zp_arr,
         'zpsys': zp_sys_arr
         }
    )

    observations.sort('time')
    return observations


def iter_lcs(obs, source, pwv_arr, z_arr, verbose=True):
    """Iterator over simulated light-curves for combination of PWV and z values

    Less memory intensive than using the default sncosmo behavior of simulating
    all light-curves in memory at once.

    Args:
        obs       (Table): Observation cadence
        source   (Source): The sncosmo source to use in the simulations
        pwv_arr (ndarray): Array of PWV values
        z_arr   (ndarray): Array of redshift values
        verbose    (bool): Show a progress bar

    Yields:
        Astropy table for each PWV and redshift
    """

    model = get_model_with_pwv(source)
    arg_iter = itertools.product(pwv_arr, z_arr)

    if verbose:
        iter_total = len(pwv_arr) * len(z_arr)
        arg_iter = tqdm(arg_iter, total=iter_total, desc='Light-Curves')

    for pwv, z in arg_iter:
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z)}

        # Some versions of sncosmo mutate arguments so we use copy to be safe
        param_list = [params.copy()]
        light_curve = sncosmo.realize_lcs(obs, model, param_list)[0]
        light_curve['zp'] = Column(light_curve['zp'], dtype=float)

        light_curve.meta = params
        yield light_curve
