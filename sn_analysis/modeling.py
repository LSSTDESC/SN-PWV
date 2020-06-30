# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module handles the simulation of SN light-curves."""

import itertools
from pathlib import Path

import numpy as np
import sncosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Column, Table
from pwv_kpno import pwv_atm
from tqdm import tqdm

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'

# From Betoule 2014
alpha = 0.141
beta = 3.101
omega_m = 0.295
abs_mb = -19.05
H0 = 70

betoule_cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)


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


def calc_x0_for_z(z, source, cosmo=betoule_cosmo, abs_mag=abs_mb, **params):
    """Determine x0 for a given redshift and model

    Args:
         z            (float): Model redshift to set
         source (Source, str): Model to use
         cosmo    (Cosmology): Cosmology to use when determining x0
         abs_mag      (float): Absolute peak magnitude of the SNe Ia
         Any other params to set for the provided `source`
    """

    model = sncosmo.Model(source)
    model.set(z=z, **params)
    model.set_source_peakabsmag(abs_mag, 'standard::b', 'AB', cosmo=cosmo)
    return model['x0']


def create_observations_table(
        phases=range(-20, 50),
        bands=('decam_g', 'decam_r', 'decam_i', 'decam_z', 'decam_y'),
        zp=25,
        zpsys='ab',
        gain=100):
    """Create an astropy table defining a uniform observation cadence for a single target

    Time values are specified in units of phase

    Args:
        phases (ndarray): Array of phase values to include
        bands  (ndarray): Array of bands to include
        zp       (float): The zero point
        zpsys    (float): The zero point system
        gain     (float): The simulated gain

    Returns:
        An astropy table
    """

    phase_arr = np.concatenate([phases for _ in bands])
    band_arr = np.concatenate([np.full_like(phases, b, dtype='U10') for b in bands])
    gain_arr = np.full_like(phase_arr, gain)
    skynoise_arr = np.zeros_like(phase_arr)
    zp_arr = np.full_like(phase_arr, zp, dtype=float)
    zp_sys_arr = np.full_like(phase_arr, zpsys, dtype='U10')

    observations = Table(
        {'time': phase_arr,
         'band': band_arr,
         'gain': gain_arr,
         'skynoise': skynoise_arr,
         'zp': zp_arr,
         'zpsys': zp_sys_arr
         },
        dtype=[float, 'U100', float, float, float, 'U100']
    )

    observations.sort('time')
    return observations


def iter_lc_simulations(obs, source, pwv_arr, z_arr, verbose=True):
    """Iterator over simulated light-curves for combination of PWV and z values

    Wrapper for the ``sncosmo.realize_lcs`` functionality that is less memory
    intensive than using the default behavior of simulating all light-curves in
    memory at once.

    .. important:: If you are not specifically looking for the the behavior of
       ``sncosmo.realize_lcs``, consider using the ``iter_lc_realizations``
       method.

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
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z, source)}

        # Some versions of sncosmo mutate arguments so we use copy to be safe
        param_list = [params.copy()]
        light_curve = sncosmo.realize_lcs(obs, model, param_list)[0]
        light_curve['zp'] = Column(light_curve['zp'], dtype=float)

        light_curve.meta = params
        yield light_curve


def iter_lc_realizations(obs, source, pwv_arr, z_arr, snr=.1, verbose=True):
    """Iterator over realized light-curves for combination of PWV and z values

    Light-curves are realized for the given parameters without any of
    the added effects from ``sncosmo.realize_lc``.

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
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z, source)}
        model.update(params)

        light_curve = Table()
        light_curve['flux'] = model.bandflux(obs['band'], obs['time'], obs['zp'], obs['zpsys'])
        light_curve['fluxerr'] = light_curve['flux'] * .1
        light_curve['zp'] = obs['zp'][0]
        light_curve['zpsys'] = obs['zpsys'][0]
        light_curve['time'] = obs['time']
        light_curve['band'] = obs['band']
        light_curve.meta = params

        yield light_curve
