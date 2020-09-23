# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module handles the simulation of SN light-curves."""

import itertools
from copy import copy
from pathlib import Path

import numpy as np
import sncosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from tqdm import tqdm

from sn_analysis.transmission import trans_for_pwv

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

    model = sncosmo.Model(source)
    model.add_effect(PWVTrans(), '', 'obs')
    model.update(params)
    return model


class TimeVariablePWV(sncosmo.Source):
    """Wrapper for SNCosmo sources that introduces time variable PWV effects"""

    def __init__(self, parent_source, pwv_func):
        """Wraps an sncosmo Source object with time variable PWV transmission effects

        Args:
            parent_source (str, Source): sncosmo source to use as a base
            pwv_func          (callable): Vectorized callable that returns PWV at given date
        """
        self.parent_source = sncosmo.get_source(parent_source)
        self.pwv_func = pwv_func

        self.name = self.parent_source.name + '_VariablePWV'
        self._wave = self.parent_source._wave
        self._phase = self.parent_source._phase

        self._parameters = self.parent_source._parameters
        self._param_names = self.parent_source._param_names
        self.param_names_latex = self.parent_source.param_names_latex

    # Todo: This is a placeholder function. Adds the actual flux calculation with PWV effects
    def _flux(self, phase, wave):
        model = sncosmo.Model(self.parent_source)
        model._parameters = self._parameters
        return model._flux(phase, wave)


###############################################################################
# For simulating light-curves
###############################################################################


def calc_x0_for_z(
        z, source, cosmo=betoule_cosmo, abs_mag=abs_mb,
        band='standard::b', magsys='AB', **params):
    """Determine x0 for a given redshift and model

    Args:
         z            (float): Model redshift to set
         source (Source, str): Model to use
         cosmo    (Cosmology): Cosmology to use when determining x0
         abs_mag      (float): Absolute peak magnitude of the SNe Ia
         band           (str): Band to set absolute magnitude in
         magsys         (str): Magnitude system to set absolute magnitude in
         Any other params to set for the provided `source`
    """

    model = sncosmo.Model(source)
    model.set(z=z, **params)
    model.set_source_peakabsmag(abs_mag, band, magsys, cosmo=cosmo)
    return model['x0']


def create_observations_table(
        phases=range(-20, 51),
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
        zpsys      (str): The zero point system
        gain     (float): The simulated gain

    Returns:
        An astropy table
    """

    phase_arr = np.concatenate([phases for _ in bands])
    band_arr = np.concatenate([np.full_like(phases, b, dtype='U1000') for b in bands])
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
        dtype=[float, 'U1000', float, float, float, 'U100']
    )

    observations.sort('time')
    return observations


def realize_lc(obs, source, snr=.05, **params):
    """Simulate a SN light-curve for given parameters

    Light-curves are simulated for the given parameters without any of
    the added effects from ``sncosmo.realize_lc``.

    Args:
        obs       (Table): Observation cadence
        source   (Source): The sncosmo source to use in the simulations
        snr       (float): Signal to noise ratio
        **params         : Values for any model parameters

    Yields:
        Astropy table for each PWV and redshift
    """

    model = get_model_with_pwv(source)
    model.update(params)

    # Set default x0 value according to assumed cosmology and the model redshift
    x0 = params.get('x0', calc_x0_for_z(model['z'], source))
    model.set(x0=x0)

    light_curve = obs[['time', 'band', 'zp', 'zpsys']]
    light_curve['flux'] = model.bandflux(obs['band'], obs['time'], obs['zp'], obs['zpsys'])
    light_curve['fluxerr'] = light_curve['flux'] / snr
    light_curve.meta = dict(zip(model.param_names, model.parameters))
    return light_curve


def simulate_lc(observations, source, params, scatter=True):
    """Simulate a SN light-curve given a set of observations.

    If ``scatter`` is ``True``, then simulated flux values include an added
    random number drawn from a Normal Distribution with a standard deviation
    equal to the error of the observation.

    Args:
        observations (Table): Table of observations.
        source       (Source): The sncosmo source to use in the simulations
        params       (dict): parameters to feed to the model for realizing the light-curve
        scatter      (bool): Add random noise to the flux values

    Returns:
        An astropy table formatted for use with sncosmo
    """

    model = sncosmo.Model(source)
    flux = model.bandflux(
        observations['band'],
        observations['time'],
        zp=observations['zp'],
        zpsys=observations['zpsys'])

    fluxerr = np.sqrt(observations['skynoise'] ** 2 + np.abs(flux) / observations['gain'])
    if scatter:
        flux = np.atleast_1d(np.random.normal(flux, fluxerr))

    data = [
        observations['time'],
        observations['band'],
        flux,
        fluxerr,
        observations['zp'],
        observations['zpsys']
    ]

    return Table(data, names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'), meta=params)


def iter_lcs(obs, source, pwv_arr, z_arr, snr=10, verbose=True):
    """Iterator over SN light-curves for combination of PWV and z values

    Light-curves are simulated for the given parameters without any of
    the added effects from ``sncosmo.realize_lc``.

    Args:
        obs       (Table): Observation cadence
        source   (Source): The sncosmo source to use in the simulations
        pwv_arr (ndarray): Array of PWV values
        z_arr   (ndarray): Array of redshift values
        snr       (float): Signal to noise ratio
        verbose    (bool): Show a progress bar

    Yields:
        Astropy table for each PWV and redshift
    """

    arg_iter = itertools.product(pwv_arr, z_arr)
    if verbose:  # pragma: no cover
        iter_total = len(pwv_arr) * len(z_arr)
        arg_iter = tqdm(arg_iter, total=iter_total, desc='Light-Curves')

    for pwv, z in arg_iter:
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z, source)}
        yield realize_lc(obs, source, snr, **params)
