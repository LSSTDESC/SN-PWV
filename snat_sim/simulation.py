"""The ``simulation`` module realizes SN Ia light-curves for a given supernova
model. The module supports supernova ``model`` objects from the ``models``
module and the ``sncosmo`` package.

Module API
----------
"""

import itertools
from copy import copy

import numpy as np
import sncosmo
from astropy.table import Table
from tqdm import tqdm

from . import constants as const


###############################################################################
# For simulating light-curves
###############################################################################


def calc_x0_for_z(
        z, source, cosmo=const.betoule_cosmo, abs_mag=const.betoule_abs_mb,
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
        {
            'time': phase_arr,
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


def realize_lc(obs, model, snr=.05, **params):
    """Simulate a SN light-curve for given parameters

    Light-curves are simulated for the given parameters without any of
    the added effects from ``sncosmo.realize_lc``.

    Args:
        obs       (Table): Observation cadence
        model     (Model): The sncosmo model to use in the simulations
        snr       (float): Signal to noise ratio
        **params         : Values for any model parameters

    Yields:
        Astropy table for each PWV and redshift
    """

    model = copy(model)
    model.update(params)

    # Set default x0 value according to assumed cosmology and the model redshift
    x0 = params.get('x0', calc_x0_for_z(model['z'], model.source))
    model.set(x0=x0)

    light_curve = obs[['time', 'band', 'zp', 'zpsys']]
    light_curve['flux'] = model.bandflux(obs['band'], obs['time'], obs['zp'], obs['zpsys'])
    light_curve['fluxerr'] = light_curve['flux'] / snr
    light_curve.meta = dict(zip(model.param_names, model.parameters))
    return light_curve


def simulate_lc(observations, model, params, scatter=True):
    """Simulate a SN light-curve given a set of observations.

    If ``scatter`` is ``True``, then simulated flux values include an added
    random number drawn from a Normal Distribution with a standard deviation
    equal to the error of the observation.

    Args:
        observations (Table): Table of observations.
        model        (Model): The sncosmo model to use in the simulations
        params        (dict): parameters to feed to the model for realizing the light-curve
        scatter       (bool): Add random noise to the flux values

    Returns:
        An astropy table formatted for use with sncosmo
    """

    model = copy(model)
    model.update(params)

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


def iter_lcs(obs, model, pwv_arr, z_arr, snr=10, verbose=True):
    """Iterator over SN light-curves for combination of PWV and z values

    Light-curves are simulated for the given parameters without any of
    the added effects from ``sncosmo.realize_lc``.

    Args:
        obs       (Table): Observation cadence
        model     (Model): The sncosmo model to use in the simulations
        pwv_arr (ndarray): Array of PWV values
        z_arr   (ndarray): Array of redshift values
        snr       (float): Signal to noise ratio
        verbose    (bool): Show a progress bar

    Yields:
        Astropy table for each PWV and redshift
    """

    model = copy(model)
    iter_total = len(pwv_arr) * len(z_arr)
    arg_iter = itertools.product(pwv_arr, z_arr)
    for pwv, z in tqdm(arg_iter, total=iter_total, desc='Light-Curves', disable=not verbose):
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z, model.source)}
        yield realize_lc(obs, model, snr, **params)