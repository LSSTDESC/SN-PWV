"""This module handles the simulation of SN light-curves.

Module API
----------
"""

import abc
import itertools
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import sncosmo
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from pwv_kpno.defaults import v1_transmission
from tqdm import tqdm

from . import constants as const

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'


def calc_airmass(time, ra, dec, lat=const.vro_latitude,
                 lon=const.vro_longitude, alt=const.vro_altitude, time_format='mjd'):
    """Calculate the airmass through which a target is observed

    Default latitude, longitude, and altitude are set to the Rubin Observatory.

    Args:
        time      (float): Time at which the target is observed
        ra        (float): Right Ascension of the target (Deg)
        dec       (float): Declination of the target (Deg)
        lat       (float): Latitude of the observer (Deg)
        lon       (float): Longitude of the observer (Deg)
        alt       (float): Altitude of the observer (m)
        time_format (str): Format of the time value (Default 'mjd')

    Returns:
        Airmass in units of Sec(z)
    """

    with warnings.catch_warnings():  # Astropy time manipulations raise annoying ERFA warnings
        warnings.filterwarnings('ignore')

        obs_time = Time(time, format=time_format)
        observer_location = EarthLocation(
            lat=lat * u.deg,
            lon=lon * u.deg,
            height=alt * u.m)

        target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        altaz = AltAz(obstime=obs_time, location=observer_location)
        return target_coord.transform_to(altaz).secz.value


###############################################################################
# For building sncosmo models with a PWV component
###############################################################################

class VariablePropagationEffect(sncosmo.PropagationEffect):
    """Similar to ``sncosmo.PropagationEffect`` class, but the ``propagate``
    method accepts a ``time`` argument.
    """

    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def propagate(self, wave, flux, time):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values
            time (ndarray): Array of time values

        Returns:
            An array of flux values after suffering propagation effects
        """

        pass  # pragma: no cover


class StaticPWVTrans(sncosmo.PropagationEffect):
    """Atmospheric propagation effect for temporally static PWV"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self):
        self._param_names = ['pwv', 'res']
        self.param_names_latex = ['PWV', 'Resolution']
        self._parameters = np.array([0., 5])

    def propagate(self, wave, flux, *args):
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


class VariablePWVTrans(VariablePropagationEffect):
    """Atmospheric propagation effect for temporally variable PWV"""

    def __init__(self, pwv_interpolator, time_format='mjd', transmission_version='v1', scale_airmass=True):
        """Time variable atmospheric transmission due to PWV

        Set ``scale_airmass`` to ``False`` if ``pwv_interpolator`` returns PWV values along the
        line of sight.

        Effect Parameters:
            ra: Target Right Ascension in degrees
            dec: Target Declination in degrees
            lat: Observer latitude in degrees (defaults to location of VRO)
            lon: Observer longitude in degrees (defaults to location of VRO)
            alt: Observer altitude in meters  (defaults to height of VRO)

        Args:
            pwv_interpolator (callable[float]): Returns PWV at zenith for a given time value
            time_format                  (str): Astropy recognized time format used by the ``pwv_interpolator``
            transmission_version         (str): Use ``v1`` of ``v2`` of the pwv_kpno transmission function
            scale_airmass               (bool): Disable airmass scaling.
        """

        # Store init arguments
        self.scale_airmass = scale_airmass
        self._time_format = time_format
        self._pwv_interpolator = pwv_interpolator

        if transmission_version == 'v1':
            from pwv_kpno.defaults import v1_transmission
            self._transmission_model = v1_transmission

        elif transmission_version == 'v2':
            from pwv_kpno.defaults import v2_transmission
            self._transmission_model = v2_transmission

        else:
            raise ValueError(f'Unidentified transmission model version: {transmission_version}')

        # Define wavelength range of propagation effect
        self._minwave = self._transmission_model.samp_wave.min()
        self._maxwave = self._transmission_model.samp_wave.max()

        # Define and store default modeling parameters
        self._param_names = ['ra', 'dec', 'lat', 'lon', 'alt', 'res']
        self.param_names_latex = [
            'Target RA', 'Target Dec', 'Observer Latitude (deg)', 'Observer Longitude (deg)',
            'Observer Altitude (m)', 'Coordinate', 'Resolution']
        self._parameters = np.array(
            [0., 0.,
             const.vro_latitude,
             const.vro_longitude,
             const.vro_altitude,
             1024, 5.])

    def airmass(self, time):
        """Return the airmass as a function of time

        Args:
            time (float, np.array): Array of time values

        Returns:
            An array of airmass values
        """

        return calc_airmass(
            time,
            ra=self['ra'],
            dec=self['dec'],
            lat=self['lat'],
            lon=self['lon'],
            alt=self['alt'],
            time_format=self._time_format)

    def calc_pwv_los(self, time):
        """Return the PWV along the line of sight for a given time

        Args:
            time (float, np.array): Array of time values

        Returns:
            An array of PWV values in mm
        """

        pwv = self._pwv_interpolator(time)
        if self.scale_airmass:
            pwv *= self.airmass(time)

        return pwv

    def propagate(self, wave, flux, time):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values
            time (ndarray): Array of time values to determine PWV for

        Returns:
            An array of flux values after suffering propagation effects
        """

        pwv = self.calc_pwv_los(time)
        transmission = self._transmission_model(pwv, np.atleast_1d(wave), self['res'])

        if np.ndim(time) == 0:  # PWV will be scalar and transmission will be a Series
            if np.ndim(flux) == 1:
                return flux * transmission

            if np.ndim(flux) == 2:
                return flux * np.atleast_2d(transmission)

        if np.ndim(time) == 1 and np.ndim(flux) == 2:  # PWV will be a vector and transmission will be a DataFrame
            return flux * transmission.values.T

        raise NotImplementedError('Could not identify how to match dimensions of Atm. model to source flux.')


class Model(sncosmo.Model):
    """Similar to ``sncosmo.Model`` class, but removes type checks from
    methods to allow duck-typing.
    """

    # Same as parent except allows duck-typing of ``effect`` arg
    def _add_effect_partial(self, effect, name, frame):
        """Like 'add effect', but don't sync parameter arrays"""

        if frame not in ['rest', 'obs', 'free']:
            raise ValueError("frame must be one of: {'rest', 'obs', 'free'}")

        self._effects.append(copy(effect))
        self._effect_names.append(name)
        self._effect_frames.append(frame)

        # for 'free' effects, add a redshift parameter
        if frame == 'free':
            self._param_names.append(name + 'z')
            self.param_names_latex.append('{\\rm ' + name + '}\\,z')

        # add all of this effect's parameters
        for param_name in effect.param_names:
            self._param_names.append(name + param_name)
            self.param_names_latex.append('{\\rm ' + name + '}\\,' + param_name)

    # Same as parent except adds support for ``VariablePropagationEffect`` effects
    def _flux(self, time, wave):
        """Array flux function."""

        a = 1. / (1. + self._parameters[0])
        phase = (time - self._parameters[1]) * a
        restwave = wave * a

        # Note that below we multiply by the scale factor to conserve
        # bolometric luminosity.
        f = a * self._source._flux(phase, restwave)

        # Pass the flux through the PropagationEffects.
        for effect, frame, zindex in zip(self._effects, self._effect_frames, self._effect_zindicies):
            if frame == 'obs':
                effect_wave = wave

            elif frame == 'rest':
                effect_wave = restwave

            else:  # frame == 'free'
                effect_a = 1. / (1. + self._parameters[zindex])
                effect_wave = wave * effect_a

            # This code block is new to the child class
            if isinstance(effect, VariablePropagationEffect):
                f = effect.propagate(effect_wave, f, time)

            else:
                f = effect.propagate(effect_wave, f)

        return f

    # Parent class copy enforces return is a parent class instance
    # Allow child classes to return copies of their own type
    def __copy__(self):
        """Like a normal shallow copy, but makes an actual copy of the
        parameter array."""

        new_model = type(self)(self.source, self.effects, self.effect_names, self._effect_frames)
        new_model.update(dict(zip(self.param_names, self.parameters)))
        return new_model


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
