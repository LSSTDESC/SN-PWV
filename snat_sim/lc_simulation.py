"""The ``lc_simulation`` module realizes light-curves for a given supernova
model. The module supports both ``snat_sim.SNModel``  and ``sncosmo.Model``
objects interchangeably.

Usage Example
-------------

Light-curves can be simulated with and without statistical noise. Both
approaches are demonstrated below.

.. doctest:: python

    >>> from snat_sim import lc_simulation, models

    >>> sn_model = models.SNModel('salt2')
    >>> sn_model.set(z=.01, x1=.5, c=-.1)

    >>> # Create a table of dates, bandpasses, gain, and skynoise values to evaluate
    >>> # the model with. Here we use the SDSS bands which come prebuilt with ``sncosmo``
    >>> band_passes = ['sdssu', 'sdssg', 'sdssr', 'sdssi']
    >>> cadence = lc_simulation.create_observations_table(bands=band_passes)

    >>> # Evaluate the model at a fixed SNR
    >>> light_curve = lc_simulation.simulate_lc_fixed_snr(cadence, sn_model, snr=5)

    >>> # Or, evaluate using statistical uncertainties determined from the gain / skynoise
    >>> light_curve = lc_simulation.simulate_lc(cadence, sn_model)


Module Docs
-----------
"""

import itertools
from copy import copy
from typing import *

import numpy as np
import sncosmo
from astropy.cosmology.core import Cosmology
from astropy.table import Table
from tqdm import tqdm

from . import constants as const
from .models import SNModel


def calc_x0_for_z(
        z: float,
        source: Union[str, sncosmo.Source],
        cosmo: Cosmology = const.betoule_cosmo,
        abs_mag: float = const.betoule_abs_mb,
        band: str = 'standard::b',
        magsys: str = 'AB',
        **params
) -> float:
    """Determine x0 for a given redshift and spectral template

    Args:
         z: Redshift to determine x0 for
         source: Spectral template to use when determining x0
         cosmo: Cosmology to use when determining x0
         abs_mag: Absolute peak magnitude of the SNe Ia
         band: Band to set absolute magnitude in
         magsys: Magnitude system to set absolute magnitude in
         Any other params to set for the provided `source`

    Returns:
        The x0 parameter for the given source and redshift
    """

    model = sncosmo.Model(source)
    model.set(z=z, **params)
    model.set_source_peakabsmag(abs_mag, band, magsys, cosmo=cosmo)
    return model['x0']


# Todo: The default gain value here makes no physical sense
def create_observations_table(
        phases: Collection[float] = range(-20, 51),
        bands: Collection[str] = ('decam_g', 'decam_r', 'decam_i', 'decam_z', 'decam_y'),
        zp: Union[int, float] = 25,
        zpsys: str = 'AB',
        gain: int = 100
) -> Table:
    """Create an astropy table defining a uniform observation cadence for a single target

    Time values are specified in units of phase by default, but can be chosen
    to reflect any time convention.

    Args:
        phases: Array of phase values to include
        bands: Array of bands to include
        zp: The zero point
        zpsys: The zero point system
        gain: The simulated gain

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


# Todo: The default gain value should match create_observations_table
def simulate_lc_fixed_snr(observations: Table, model: SNModel, snr: float = .05, **params) -> Table:
    """Simulate a SN light-curve with a fixed SNR given a set of observations

    The ``obs`` table is expected to have columns for 'time', 'band', 'zp',
    and 'zpsys'.

    Args:
        observations: Table outlining the observation cadence
        model: Supernova model to evaluate
        snr: Signal to noise ratio
        **params: Values for any model parameters

    Returns:
        An astropy table formatted for use with sncosmo
    """

    model = copy(model)
    model.update(params)

    # Set default x0 value according to assumed cosmology and the model redshift
    x0 = params.get('x0', calc_x0_for_z(model['z'], model.source))
    model.set(x0=x0)

    light_curve = observations[['time', 'band', 'zp', 'zpsys']]
    light_curve['flux'] = model.bandflux(observations['band'], observations['time'], observations['zp'], observations['zpsys'])
    light_curve['fluxerr'] = light_curve['flux'] / snr
    light_curve.meta = dict(zip(model.param_names, model.parameters))
    return light_curve


def iter_lcs_fixed_snr(
        obs: Table,
        model: SNModel,
        pwv_arr: np.array,
        z_arr: np.array,
        snr: Union[float, int] = 10,
        verbose: bool = True
) -> Iterator[Table]:
    """Iterator over SN light-curves for combination of PWV and z values

    Args:
        obs: Observation cadence
        model: The sncosmo model to use in the simulations
        pwv_arr: Array of PWV values
        z_arr: Array of redshift values
        snr: Signal to noise ratio
        verbose: Show a progress bar

    Yields:
        An Astropy table for each PWV and redshift
    """

    model = copy(model)
    iter_total = len(pwv_arr) * len(z_arr)
    arg_iter = itertools.product(pwv_arr, z_arr)
    for pwv, z in tqdm(arg_iter, total=iter_total, desc='Light-Curves', disable=not verbose):
        params = {'t0': 0.0, 'pwv': pwv, 'z': z, 'x0': calc_x0_for_z(z, model.source)}
        yield simulate_lc_fixed_snr(obs, model, snr, **params)


def simulate_lc(observations: Table, model: SNModel, params: Dict[str, float] = None, scatter: bool = True) -> Table:
    """Simulate a SN light-curve given a set of observations

    If ``scatter`` is ``True``, then simulated flux values include an added
    random number drawn from a Normal Distribution with a standard deviation
    equal to the error of the observation.

    The ``observations`` table is expected to have columns for 'time', 'band',
    'zp', 'zpsys', 'skynoise', and 'gain'.

    Args:
        observations: Table of observations.
        model: The sncosmo model to use in the simulations
        params: parameters to feed to the model for realizing the light-curve
        scatter: Add random noise to the flux values

    Returns:
        An astropy table formatted for use with sncosmo
    """

    if params is None:
        params = dict()

    model = copy(model)
    for p in model.param_names:
        model[p] = params.get(p, model[p])

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
