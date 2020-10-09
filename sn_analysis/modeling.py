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
from pwv_kpno.defaults import v1_transmission
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


class PWVTrans(sncosmo.PropagationEffect):
    """Atmospheric PWV propagation effect for sncosmo"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self):
        self._param_names = ['pwv', 'res']
        self.param_names_latex = ['PWV', 'Resolution']
        self._parameters = np.array([0., 5])

    def propagate(self, wave, flux):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values

        Returns:
            An array of flux values after suffering propagation effects
        """

        # The class guarantees PWV is a scalar, so the transmission is 1D
        pwv, res = self.parameters
        transmission = v1_transmission(pwv, wave, res)

        # The flux is 2D, so we do a quick cast
        return flux * transmission.values[None, :]


class PWVVariableModel(sncosmo.Model):
    """Clone of the ``sncosmo.Model`` class but with time variable PWV effects"""

    def __init__(self, source, pwv_func, effects=None, effect_names=None, effect_frames=None):
        """An observer-frame supernova model that incorporates time variable PWV transmission

        .. important:: Arguments for the ``t0`` parameter and the ``pwv_func``
           callable must be in the same units.

        Args:
            source (Source, str): The model for the spectral evolution of the source.
            pwv_func  (callable): Function that returns PWV for a given time values. Must support vectors
            effects       (list): list of ``sncosmo.PropagationEffect``
            effect_names  (list): Names of each propagation effect (same length as `effects`).
            effect_frames (list): The frame that each effect is in (same length as `effects`).
                Must be one of {'rest', 'obs'}."""


        self._pwv_func = pwv_func

        # Set parameter names, initial values (inital values set to zero)
        self._param_names = ['z', 't0', 'res']
        self.param_names_latex = ['z', 't_0', 'Resolution']
        self._parameters = np.array([0, 0, 5])

        # Set source and add source parameter names
        self._source = sncosmo.get_source(source, copy=True)
        self._param_names.extend(self._source.param_names)
        self.param_names_latex.extend(self._source.param_names_latex)

        # Add PropagationEffects
        self._effects = []
        self._effect_names = []
        self._effect_frames = []
        if (effects is not None or effect_names is not None or
                effect_frames is not None):
            try:
                same_length = (len(effects) == len(effect_names) and
                               len(effects) == len(effect_frames))
            except TypeError:
                raise TypeError('effects, effect_names, and effect_frames '
                                'should all be iterables.')
            if not same_length:
                raise ValueError('effects, effect_names and effect_frames '
                                 'must have matching lengths')

            for effect, name, frame in zip(effects, effect_names,
                                           effect_frames):
                self._add_effect_partial(effect, name, frame)

        # sync
        self._sync_parameter_arrays()
        self._update_description()

    def _sync_parameter_arrays(self):
        """Synchronize parameter names and parameter arrays between
        the aggregated parameters and those of the individual source and
        effects.
        """

        num_additional_params = 3

        # save a reference to old parameter values, in case there are
        # effect redshifts that have been set.
        old_parameters = self._parameters

        # Calculate total length of model's parameter array
        l = num_additional_params + len(self._source._parameters)
        for effect, frame in zip(self._effects, self._effect_frames):
            l += (frame == 'free') + len(effect._parameters)

        # allocate new array (zeros so that new 'free' effects redshifts
        # initialize to 0)
        self._parameters = np.zeros(l, dtype=np.float)

        # copy old parameters: we do this to make sure we copy
        # non-default values of any parameters that the model alone
        # holds, such as z, t0 and effect redshifts.
        self._parameters[0:len(old_parameters)] = old_parameters

        # cross-reference source's parameters
        pos = num_additional_params
        l = len(self._source._parameters)
        self._parameters[pos:pos+l] = self._source._parameters  # copy
        self._source._parameters = self._parameters[pos:pos+l]  # reference
        pos += l

        # initialize a list of ints that keeps track of where the redshift
        # parameter of each effect is. Value is 0 if effect_frame is not 'free'
        self._effect_zindicies = []

        # for each effect, cross-reference the effect's parameters
        for i in range(len(self._effects)):
            effect = self._effects[i]

            # for 'free' effects, add a redshift parameter
            if self._effect_frames[i] == 'free':
                self._effect_zindicies.append(pos)
                pos += 1
            else:
                self._effect_zindicies.append(-1)

            # add all of this effect's parameters
            l = len(effect._parameters)
            self._parameters[pos:pos+l] = effect._parameters  # copy
            effect._parameters = self._parameters[pos:pos+l]  # reference
            pos += l

    def calc_pwv_los(self, time):
        """Return the PWV along the line of sight for a given time

        Args:
            time (float, np.array): Array of time values

        Returns:
            An array of PWV values in mm
        """

        # Todo: Add RA and Dec parameters. Use them to scale PWV to the appropriate airmass
        # Extend tests appropriately
        return self._pwv_func(time)

    def _flux(self, time, wave):

        pwv = self.calc_pwv_los(time)
        flux = super()._flux(time, wave)
        transmission = v1_transmission(pwv, wave, self['res'])

        if np.ndim(time) == 0:  # PWV will be scalar and transmission will be a Series
            if np.ndim(flux) == 1:
                return flux * transmission

            if np.ndim(flux) == 2:
                return flux * np.atleast_2d(transmission)

        if np.ndim(time) == 1 and np.ndim(flux) == 2:  # PWV will be a vector and transmission will be a DataFrame
            return flux * transmission.values.T

        raise NotImplementedError('Could not identify how to match dimensions of Atm. model to source flux.')

    def __copy__(self):
        """Like a normal shallow copy, but makes an actual copy of the
        parameter array."""

        new_model = self.__new__(self.__class__)
        for key, val in self.__dict__.items():
            new_model.__dict__[key] = val

        new_model._parameters = copy(self._parameters)
        return new_model


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
