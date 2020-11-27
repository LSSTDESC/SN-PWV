"""The ``models`` module defines classes that act as models for different
physical phenomena. This includes SNe Ia light-curves, the propagation of
light through atmospheric water vapor (with and without variation in time),
and the seasonal variation of water vapor vs time.

Usage Example
-------------

To ensure backwards compatibility and ease of use, supernovae modeling with the
``snat_sim`` package follows the
`same design <https://sncosmo.readthedocs.io/en/stable/models.html>`_
as the ``sncosmo`` package. Models are instantiated for a given spectral
template and various propagation effects can be added to the model. In the
following example, atmospheric propagation effects due to precipitable water
vapor are added to a Salt2 supernova model.

.. doctest:: python

   >>> from snat_sim import models

   >>> # Create a supernova model
   >>> supernova_model = models.SNModel('salt2')

   >>> # Create a model for the atmosphere
   >>> atm_transmission = models.StaticPWVTrans()
   >>> atm_transmission.set(pwv=4)
   >>> supernova_model.add_effect(effect=atm_transmission, name='Atmosphere', frame='obs')

Module Docs
-----------
"""

import abc
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import sncosmo
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from pwv_kpno.defaults import v1_transmission
from pwv_kpno.transmission import calc_pwv_eff
from scipy.interpolate import RegularGridInterpolator

from . import constants as const
from . import time_series_utils as tsu
from .cache_utils import fast_cache

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'


class FixedResTransmission:
    """Models atmospheric transmission due to PWV at a fixed resolution"""

    def __init__(self, res=None, cache_trans=False):
        """Instantiate a PWV transmission model at the given resolution

        Transmission values are determined using the ``v1_transmission`` model
        from the ``pwv_kpno`` package.

        Args:
            res       (float): Resolution to bin the atmospheric model to
            cache_trans (int): Optionally cache transmission calculations up to given number of bytes
        """

        self.norm_pwv = 2
        self.eff_exp = 0.6
        self.samp_pwv = np.arange(0, 15, .1)
        self.samp_wave = v1_transmission.samp_wave
        self.samp_transmission = v1_transmission(
            pwv=self.samp_pwv,
            wave=self.samp_wave,
            res=res).values.T

        self._interpolator = RegularGridInterpolator(
            points=(calc_pwv_eff(self.samp_pwv), self.samp_wave), values=self.samp_transmission)

        if cache_trans:
            self.calc_transmission = fast_cache('pwv', 'wave', cache_size=cache_trans)(self.calc_transmission)

    def calc_transmission(self, pwv, wave=None):
        """Evaluate transmission model at given wavelengths

        Returns a ``Series`` object if ``pwv`` is a scalar, and a ``DataFrame``
        object if ``pwv`` is an array. Wavelengths are expected in angstroms.

        Args:
            pwv (float, Iterable[float]): Line of sight PWV to interpolate for
            wave    (ndarray, DataFrame): Wavelengths to evaluate transmission

        Returns:
            The interpolated transmission at the given wavelengths / resolution
        """

        wave = self.samp_wave if wave is None else wave
        pwv_eff = calc_pwv_eff(pwv, norm_pwv=self.norm_pwv, eff_exp=self.eff_exp)

        if np.isscalar(pwv_eff):
            xi = [[pwv_eff, w] for w in wave]
            return pd.Series(self._interpolator(xi), index=wave, name=f'{float(np.round(pwv, 4))} mm')

        else:
            xi = [[[pwv_val, w] for pwv_val in pwv_eff] for w in wave]
            names = map('{} mm'.format, np.round(pwv, 4).astype(float))
            return pd.DataFrame(self._interpolator(xi), columns=names)


class PWVModel:
    """Interpolation model for the PWV at a given point of the year"""

    def __init__(self, pwv_series, cache_pwv_los=False):
        """Build a model for time variable PWV by drawing from a given PWV time series

        Args:
            pwv_series (Series): PWV values with a datetime index
            cache_pwv_los (int): Optionally cache line of sight PWV values up to given number of bytes
        """

        self.pwv_model_data = tsu.periodic_interpolation(tsu.resample_data_across_year(pwv_series))
        self.pwv_model_data.index = tsu.datetime_to_sec_in_year(self.pwv_model_data.index)

        if cache_pwv_los:
            self.pwv_los = fast_cache('date', cache_size=cache_pwv_los)(self.pwv_los)

    @staticmethod
    def from_suominet_receiver(receiver, year, supp_years=None, cache_pwv_los=False):
        """Construct a ``PWVModel`` instance using data from a SuomiNet receiver

        Args:
            receiver (pwv_kpno.GPSReceiver): GPS receiver to access data from
            year                    (float): Year to use data from when building the model
            supp_years      (List[Numeric]): Years to supplement data with when missing from ``year``
            cache_pwv_los             (int): Set ``cache_pwv_los`` for the returned ``PWVModel`` object

        Returns:
            An interpolation function that accepts ``date`` and ``format`` arguments
        """

        weather_data = receiver.weather_data().PWV
        supp_data = tsu.supplemented_data(weather_data, year, supp_years)
        return PWVModel(supp_data, cache_pwv_los=cache_pwv_los)

    @staticmethod
    def calc_airmass(time, ra, dec, lat=const.vro_latitude,
                     lon=const.vro_longitude, alt=const.vro_altitude,
                     time_format='mjd'):
        """Calculate the airmass through which a target is observed

        Default latitude, longitude, and altitude are set to the Rubin
        Observatory.

        Args:
            time (float, List[float]): Time at which the target is observed
            ra        (float): Right Ascension of the target (Deg)
            dec       (float): Declination of the target (Deg)
            lat       (float): Latitude of the observer (Deg)
            lon       (float): Longitude of the observer (Deg)
            alt       (float): Altitude of the observer (m)
            time_format (str): Astropy supported format of the time value (Default: 'mjd')

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

    def pwv_zenith(self, date, time_format='mjd'):
        """Interpolate the PWV at zenith as a function of time

        The ``time_format`` argument can be set to ``None`` when passing datetime
        objects instead of numerical values for ``date``.

        Args:
            date (float, List[float]): The date to interpolate PWV for
            time_format         (str): Astropy supported format of the time value (Default: 'mjd')

        Returns:
            The PWV at zenith for the given date(s)
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_as_datetime = Time(date, format=time_format).to_datetime()
            x_in_seconds = tsu.datetime_to_sec_in_year(x_as_datetime)

        return np.interp(
            x=x_in_seconds,
            xp=self.pwv_model_data.index,
            fp=self.pwv_model_data.values
        )

    def pwv_los(self, date, ra, dec, lat=const.vro_latitude,
                lon=const.vro_longitude, alt=const.vro_altitude,
                time_format='mjd'):
        """Interpolate the PWV along the line of sight as a function of time

        The ``time_format`` argument can be set to ``None`` when passing datetime
        objects instead of numerical values for ``date``.

        Args:
            date (float, List[float]): Time at which the target is observed
            ra                (float): Right Ascension of the target (Deg)
            dec               (float): Declination of the target (Deg)
            lat               (float): Latitude of the observer (Deg)
            lon               (float): Longitude of the observer (Deg)
            alt               (float): Altitude of the observer (m)
            time_format         (str): Astropy supported format of the time value (Default: 'mjd')
        """

        return (self.pwv_zenith(date, time_format) *
                self.calc_airmass(date, ra, dec, lat, lon, alt, time_format))


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
    """Atmospheric propagation effect for time static PWV"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self, transmission_res=5, cache_trans=False):
        """Time independent atmospheric transmission due to PWV

        Setting the ``transmission_res`` argument to ``None`` results in the
        highest available transmission model available.

        Effect Parameters:
            pwv: Atmospheric concentration of PWV along line of sight in mm
            cache_trans (int): Optionally cache transmission calculations up to given number of bytes

        Args:
            transmission_res (float): Reduce the underlying transmission model by binning to the given resolution
        """

        self._transmission_res = transmission_res
        self._param_names = ['pwv']
        self.param_names_latex = ['PWV']
        self._parameters = np.array([0.])
        self._transmission_model = FixedResTransmission(transmission_res, cache_trans=cache_trans)

    @property
    def transmission_res(self):
        """Resolution used when binning the underlying atmospheric transmission model"""

        return self._transmission_res

    def propagate(self, wave, flux, *args):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): A 1D array of wavelength values
            flux (ndarray): An array of flux values

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        # The class guarantees PWV is a scalar, so ``transmission`` is 1D
        transmission = self._transmission_model.calc_transmission(self.parameters[0], wave)

        # ``flux`` is 2D, so we do a quick cast
        return flux * transmission.values[None, :]


class VariablePWVTrans(VariablePropagationEffect, StaticPWVTrans):
    """Atmospheric propagation effect for temporally variable PWV"""

    def __init__(self, pwv_model, time_format='mjd', transmission_res=5., cache_trans=False):
        """Time variable atmospheric transmission due to PWV

        Setting the ``transmission_res`` argument to ``None`` results in the
        highest available transmission model available.

        Effect Parameters:
            ra: Target Right Ascension in degrees
            dec: Target Declination in degrees
            lat: Observer latitude in degrees (defaults to location of VRO)
            lon: Observer longitude in degrees (defaults to location of VRO)
            alt: Observer altitude in meters  (defaults to height of VRO)

        Args:
            pwv_model       (PWVModel): Returns PWV at zenith for a given time value and time format
            time_format          (str): Astropy recognized time format used by the ``pwv_interpolator``
            transmission_res   (float): Reduce the underlying transmission model by binning to the given resolution
            cache_trans          (int): Optionally cache transmission calculations up to given number of bytes
        """

        # Create atmospheric transmission model
        super().__init__(transmission_res=transmission_res, cache_trans=cache_trans)

        self._time_format = time_format
        self._pwv_model = pwv_model

        # Define wavelength range of propagation effect
        self._minwave = self._transmission_model.samp_wave.min()
        self._maxwave = self._transmission_model.samp_wave.max()

        # Define and store default modeling parameters
        self._param_names = ['ra', 'dec', 'lat', 'lon', 'alt']
        self.param_names_latex = [
            'Target RA', 'Target Dec',
            'Observer Latitude (deg)', 'Observer Longitude (deg)', 'Observer Altitude (m)']

        self._parameters = np.array([0., 0., const.vro_latitude, const.vro_longitude, const.vro_altitude])

    def propagate(self, wave, flux, time):
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values
            time (ndarray): Array of time values to determine PWV for

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        pwv = self._pwv_model.pwv_los(
            date=time,
            ra=self['ra'],
            dec=self['dec'],
            lat=self['lat'],
            lon=self['lon'],
            alt=self['alt'],
            time_format=self._time_format)

        transmission = self._transmission_model.calc_transmission(pwv=pwv, wave=np.atleast_1d(wave))

        if np.ndim(time) == 0:  # PWV will be scalar and transmission will be a Series
            if np.ndim(flux) == 1:
                return flux * transmission

            if np.ndim(flux) == 2:
                return flux * np.atleast_2d(transmission)

        if np.ndim(time) == 1 and np.ndim(flux) == 2:  # PWV will be a vector and transmission will be a DataFrame
            return flux * transmission.values.T

        raise NotImplementedError('Could not identify how to match dimensions of atm. model to source flux.')


class SNModel(sncosmo.Model):
    """Similar to ``sncosmo.SNModel`` class, but removes type checks from
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
