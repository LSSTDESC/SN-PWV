"""The ``models`` module defines classes that act as models for different
physical phenomena. This includes SNe Ia light-curves, the propagation of
light through atmospheric water vapor (with and without variation in time),
and the seasonal variation of water vapor vs time.

Model Summaries
---------------

A summary of the available models is provided below:

.. autosummary::
   :nosignatures:

   FixedResTransmission
   PWVModel
   SNModel

Supernova models (``SNModel``) are designed to closely resemble the behavior
of the ``sncosmo`` package. However, unlike ``sncosmo.Model`` objects, the
``snat_sim.SNModel`` class provides support for propogation effects that vary
with time. A summary of propagation effects provided by the ``snat_sim``
package is listed below:

.. autosummary::
   :nosignatures:

   StaticPWVTrans
   SeasonalPWVTrans
   VariablePWVTrans

Usage Example
-------------

To ensure backwards compatibility and ease of use, supernovae modeling with the
``snat_sim`` package follows the same
`design pattern <https://sncosmo.readthedocs.io/en/stable/models.html>`_
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

from __future__ import annotations

import abc
import warnings
from copy import copy
from datetime import datetime
from typing import *

import joblib
import numpy as np
import pandas as pd
import sncosmo
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from pwv_kpno.defaults import v1_transmission
from pwv_kpno.gps_pwv import GPSReceiver
from pwv_kpno.transmission import calc_pwv_eff
from pytz import utc
from scipy.interpolate import RegularGridInterpolator

from . import constants as const
from .utils import time_series as tsu
from snat_sim.utils.caching import numpy_cache
from ._data_paths import data_paths

# Todo: These were picked ad-hock and are likely too big.
#  They should be set to a reasonable number further along in development
PWV_CACHE_SIZE = 500_000
TRANSMISSION_CACHE_SIZE = 500_00
AIRMASS_CACHE_SIZE = 250_000

NumpyVar = TypeVar('NumpyVar', float, np.ndarray)


###############################################################################
# Core models for physical phenomena
###############################################################################

class FixedResTransmission:
    """Models atmospheric transmission due to PWV at a fixed resolution"""

    def __init__(self, resolution: float = None) -> None:
        """Instantiate a PWV transmission model at the given resolution

        Transmission values are determined using the ``v1_transmission`` model
        from the ``pwv_kpno`` package.

        Args:
            resolution: Resolution to bin the atmospheric model to
        """

        self.norm_pwv = 2
        self.eff_exp = 0.6
        self.samp_pwv = np.arange(0, 60.25, .25)
        self.samp_wave = v1_transmission.samp_wave
        self.samp_transmission = v1_transmission(
            pwv=self.samp_pwv,
            wave=self.samp_wave,
            res=resolution).values.T

        self._interpolator = RegularGridInterpolator(
            points=(calc_pwv_eff(self.samp_pwv), self.samp_wave), values=self.samp_transmission)

        self.calc_transmission = numpy_cache('pwv', 'wave', cache_size=TRANSMISSION_CACHE_SIZE)(self.calc_transmission)

    @overload
    def calc_transmission(self, pwv: float, wave: Optional[np.array] = None) -> pd.Series:
        ...

    @overload
    def calc_transmission(self, pwv: Collection[float], wave: Optional[np.ndarray] = None) -> pd.DataFrame:
        ...

    def calc_transmission(self, pwv, wave=None):
        """Evaluate transmission model at given wavelengths

        Returns a ``Series`` object if ``pwv`` is a scalar, and a ``DataFrame``
        object if ``pwv`` is an array. Wavelengths are expected in angstroms.

        Args:
            pwv: Line of sight PWV to interpolate for
            wave: Wavelengths to evaluate transmission (Defaults to ``samp_wave`` attribute)

        Returns:
            The interpolated transmission at the given wavelengths / resolution
        """

        wave = self.samp_wave if wave is None else wave
        pwv_eff = calc_pwv_eff(pwv, norm_pwv=self.norm_pwv, eff_exp=self.eff_exp)

        if np.isscalar(pwv_eff):
            xi = [[pwv_eff, w] for w in wave]
            return pd.Series(self._interpolator(xi), index=wave, name=f'{float(np.round(pwv, 4))} mm')

        else:
            # Equivalent to [[[pwv_val, w] for pwv_val in pwv_eff] for w in wave]
            xi = np.empty((len(wave), len(pwv_eff), 2))
            xi[:, :, 0] = pwv_eff
            xi[:, :, 1] = np.array(wave)[:, None]

            names = map('{} mm'.format, np.round(pwv, 4).astype(float))
            return pd.DataFrame(self._interpolator(xi), columns=names)


class PWVModel:
    """Model for interpolating the atmospheric water vapor at a given time and time"""

    def __init__(self, pwv_series: pd.Series) -> None:
        """Build a model for time variable PWV by drawing from a given PWV time series

        Args:
            pwv_series: PWV values with a datetime index
        """

        self.pwv_model_data = tsu.periodic_interpolation(tsu.resample_data_across_year(pwv_series))
        self.pwv_model_data.index = tsu.datetime_to_sec_in_year(self.pwv_model_data.index)

        self.pwv_los = numpy_cache('time', cache_size=PWV_CACHE_SIZE)(self.pwv_los)

        memory = joblib.Memory(str(data_paths.data_dir), verbose=0, bytes_limit=AIRMASS_CACHE_SIZE)
        self.calc_airmass = memory.cache(self.calc_airmass)

    @staticmethod
    def from_suominet_receiver(receiver: GPSReceiver, year: int, supp_years: Collection[int] = None) -> PWVModel:
        """Construct a ``PWVModel`` instance using data from a SuomiNet receiver

        Args:
            receiver: GPS receiver to access data from
            year: Year to use data from when building the model
            supp_years: Years to supplement data with when missing from ``year``

        Returns:
            An interpolation function that accepts ``time`` and ``format`` arguments
        """

        all_years = [year]
        if supp_years:
            all_years.extend(supp_years)

        receiver.download_available_data(all_years)

        weather_data = receiver.weather_data().PWV
        supp_data = tsu.supplemented_data(weather_data, year, supp_years)
        return PWVModel(supp_data)

    @overload
    def calc_airmass(
            self, time: float, ra: float, dec: float, lat: float, lon: float, alt: float, time_format: str) -> float:
        ...

    @overload
    def calc_airmass(
            self, time: List[float], ra: float, dec: float, lat: float, lon: float, alt: float, time_format: str) -> np.array:
        ...

    @staticmethod
    def calc_airmass(
            time, ra, dec, lat=const.vro_latitude, lon=const.vro_longitude, alt=const.vro_altitude, time_format='mjd'):
        """Calculate the airmass through which a target is observed

        Default latitude, longitude, and altitude are set to the Rubin
        Observatory.

        Args:
            time: Time at which the target is observed
            ra: Right Ascension of the target (Deg)
            dec: Declination of the target (Deg)
            lat: Latitude of the observer (Deg)
            lon: Longitude of the observer (Deg)
            alt: Altitude of the observer (m)
            time_format: Astropy supported format of the time value (Default: 'mjd')

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

    @overload
    def pwv_zenith(self, time: float, time_format: str) -> float:
        ...

    @overload
    def pwv_zenith(self, time: List[float], time_format: str) -> np.array:
        ...

    def pwv_zenith(self, time, time_format='mjd'):
        """Interpolate the PWV at zenith as a function of time

        The ``time_format`` argument can be set to ``None`` when passing datetime
        objects instead of numerical values for ``time``.

        Args:
            time: The time to interpolate PWV for
            time_format: Astropy supported format of the time value (Default: 'mjd')

        Returns:
            The PWV at zenith for the given time(s)
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_as_datetime = Time(time, format=time_format).to_datetime()
            x_in_seconds = tsu.datetime_to_sec_in_year(x_as_datetime)

        return np.interp(
            x=x_in_seconds,
            xp=self.pwv_model_data.index,
            fp=self.pwv_model_data.values
        )

    @overload
    def pwv_los(
            self, time: float, ra: float, dec: float, lat: float, lon: float, alt: float, time_format: str) -> float:
        ...

    @overload
    def pwv_los(
            self, time: List[float], ra: float, dec: float, lat: float, lon: float, alt: float,
            time_format: str) -> np.array:
        ...

    def pwv_los(
            self,
            time: Union[float, List[float]],
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude, alt=const.vro_altitude,
            time_format: str = 'mjd'
    ) -> Union[float, np.array]:
        """Interpolate the PWV along the line of sight as a function of time

        The ``time_format`` argument can be set to ``None`` when passing datetime
        objects instead of numerical values for ``time``.

        Args:
            time: Time at which the target is observed
            ra: Right Ascension of the target (Deg)
            dec: Declination of the target (Deg)
            lat: Latitude of the observer (Deg)
            lon: Longitude of the observer (Deg)
            alt: Altitude of the observer (m)
            time_format: Astropy supported format of the time value (Default: 'mjd')

        Returns:
            The PWV concentration along the line of sight to the target
        """

        return (self.pwv_zenith(time, time_format) *
                self.calc_airmass(time, ra, dec, lat, lon, alt, time_format))

    def seasonal_averages(self) -> Dict[str, float]:
        """Calculate the average PWV in each season

        Assumes seasons based on equinox and solstice dates in the year 2020.

        Returns:
            A dictionary with the average PWV in each season (in mm)
        """

        # Rough estimates for the start of each season
        spring = tsu.datetime_to_sec_in_year(datetime(2020, 3, 20, tzinfo=utc))
        summer = tsu.datetime_to_sec_in_year(datetime(2020, 6, 21, tzinfo=utc))
        fall = tsu.datetime_to_sec_in_year(datetime(2020, 9, 22, tzinfo=utc))
        winter = tsu.datetime_to_sec_in_year(datetime(2020, 12, 21, tzinfo=utc))

        # Separate PWV data based on season
        winter_pwv = self.pwv_model_data[(self.pwv_model_data.index < spring) | (self.pwv_model_data.index > winter)]
        spring_pwv = self.pwv_model_data[(self.pwv_model_data.index > spring) & (self.pwv_model_data.index < summer)]
        summer_pwv = self.pwv_model_data[(self.pwv_model_data.index > summer) & (self.pwv_model_data.index < fall)]
        fall_pwv = self.pwv_model_data[(self.pwv_model_data.index > fall) & (self.pwv_model_data.index < winter)]

        return {
            'winter': winter_pwv.mean(),
            'spring': spring_pwv.mean(),
            'summer': summer_pwv.mean(),
            'fall': fall_pwv.mean()
        }


class SNModel(sncosmo.Model):
    """An observer-frame supernova model composed of a Source and zero or more effects"""

    # Same as parent except allows duck-typing of ``effect`` arg
    def _add_effect_partial(self, effect, name, frame) -> None:
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
    def _flux(self, time, wave) -> np.ndarray:
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
    def __copy__(self) -> SNModel:
        """Like a normal shallow copy, but makes an actual copy of the
        parameter array."""

        new_model = type(self)(self.source, self.effects, self.effect_names, self._effect_frames)
        new_model.update(dict(zip(self.param_names, self.parameters)))
        return new_model


###############################################################################
# Propagation effects
###############################################################################


class VariablePropagationEffect(sncosmo.PropagationEffect):
    """Base class for propogation effects that vary with time

    Similar to ``sncosmo.PropagationEffect`` class, but the ``propagate``
    method accepts a ``time`` argument.
    """

    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def propagate(self, wave: np.ndarray, flux: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Propagate the flux through the atmosphere

        Args:
            wave: An array of wavelength values
            flux: An array of flux values
            time: Array of time values

        Returns:
            An array of flux values after suffering propagation effects
        """

        pass  # pragma: no cover


class StaticPWVTrans(sncosmo.PropagationEffect):
    """Propagation effect for the atmospheric absorption of light due to time static PWV"""

    _minwave = 3000.0
    _maxwave = 12000.0

    def __init__(self, transmission_res: float = 5) -> None:
        """Time independent atmospheric transmission due to PWV

        Setting the ``transmission_res`` argument to ``None`` results in the
        highest available transmission model available.

        Effect Parameters:
            pwv: Atmospheric concentration of PWV along line of sight in mm

        Args:
            transmission_res (float): Reduce the underlying transmission model by binning to the given resolution
        """

        self._transmission_res = transmission_res
        self._param_names = ['pwv']
        self.param_names_latex = ['PWV']
        self._parameters = np.array([0.])
        self._transmission_model = FixedResTransmission(transmission_res)

    @property
    def transmission_res(self) -> float:
        """Resolution used when binning the underlying atmospheric transmission model"""

        return self._transmission_res

    def propagate(self, wave: np.ndarray, flux: np.ndarray, *args) -> np.ndarray:
        """Propagate the flux through the atmosphere

        Args:
            wave: A 1D array of wavelength values
            flux: An array of flux values

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        # The class guarantees PWV is a scalar, so ``transmission`` is 1D
        transmission = self._transmission_model.calc_transmission(self.parameters[0], wave)

        # ``flux`` is 2D, so we do a quick cast
        return flux * transmission.values[None, :]


class VariablePWVTrans(VariablePropagationEffect, StaticPWVTrans):
    """Propagation effect for the atmospheric absorption of light due to time variable PWV"""

    def __init__(self, pwv_model: PWVModel, time_format: str = 'mjd', transmission_res: float = 5.) -> None:
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
            pwv_model: Returns PWV at zenith for a given time value and time format
            time_format: Astropy recognized time format used by the ``pwv_interpolator``
            transmission_res: Reduce the underlying transmission model by binning to the given resolution
        """

        # Create atmospheric transmission model
        super().__init__(transmission_res=transmission_res)

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

    def _apply_propagation(self, time: Union[float, np.ndarray], flux: np.ndarray,
                           transmission: np.ndarray) -> np.ndarray:
        """Apply an atmospheric transmission to flux values

        Args:
            time: Time values for the sampled flux and transmission values
            flux: Array of flux values
            transmission: Array of sampled transmission values
        """

        if np.ndim(time) == 0:  # PWV will be scalar and transmission will be a Series
            if np.ndim(flux) == 1:
                return flux * transmission

            if np.ndim(flux) == 2:
                return flux * np.atleast_2d(transmission)

        if np.ndim(time) == 1 and np.ndim(flux) == 2:  # PWV will be a vector and transmission will be a DataFrame
            return flux * transmission.values.T

        raise NotImplementedError('Could not identify how to match dimensions of atm. model to source flux.')

    def assumed_pwv(self, time: NumpyVar) -> NumpyVar:
        """The PWV concentration used by the propagation effect at a given time

        Args:
            time (float, array): Time to get the PWV concentration for

        Returns:
            An array of PWV values in units of mm
        """

        return self._pwv_model.pwv_los(
            time,
            ra=self['ra'],
            dec=self['dec'],
            lat=self['lat'],
            lon=self['lon'],
            alt=self['alt'],
            time_format=self._time_format)

    def propagate(self, wave: np.ndarray, flux: np.ndarray, time: Union[float, np.ndarray]) -> np.ndarray:
        """Propagate the flux through the atmosphere

        Args:
            wave: An array of wavelength values
            flux: An array of flux values
            time: Array of time values to determine PWV for

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        pwv = self.assumed_pwv(time)
        transmission = self._transmission_model.calc_transmission(pwv, np.atleast_1d(wave))
        return self._apply_propagation(time, flux, transmission)


class SeasonalPWVTrans(VariablePWVTrans):
    """Atmospheric propagation effect for a fixed PWV concentration per-season"""

    def assumed_pwv(self, time: NumpyVar) -> NumpyVar:
        """The PWV concentration used by the propagation effect at a given time

        Args:
            time (float, array): Time to get the PWV concentration for

        Returns:
            An array of PWV values in units of mm
        """

        # Convert time values to their corresponding season
        datetime_objects = Time(time, format=self._time_format).to_datetime()
        seasons = tsu.datetime_to_season(datetime_objects)

        # Get the average PWV for each season
        avg_pwv_per_season = self._pwv_model.seasonal_averages()
        return np.array([avg_pwv_per_season[season] for season in seasons])

    def propagate(self, wave: np.ndarray, flux: np.ndarray, time: Union[float, np.ndarray]) -> np.ndarray:
        """Propagate the flux through the atmosphere

        Args:
            wave (ndarray): An array of wavelength values
            flux (ndarray): An array of flux values
            time (ndarray): Array of time values to determine PWV for

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        pwv = self.assumed_pwv(time)
        transmission = self._transmission_model.calc_transmission(pwv, np.atleast_1d(wave))
        return self._apply_propagation(time, flux, transmission)
