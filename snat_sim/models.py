"""The ``models`` module defines classes for modeling different
physical phenomena. This includes SNe Ia light-curves, the propagation of
light through atmospheric water vapor (with and without variation in time),
and the seasonal variation of precipitable water vapor over time.

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


To simulate a light-curve, you must first establish the desired light-curve
cadence (i.e., how the light-curve should be sampled in time).

.. doctest:: python

    >>> cadence = models.ObservedCadence(
    ...     obs_times=[-1, 0, 1],
    ...     bands=['sdssr', 'sdssr', 'sdssr'],
    ...     zp=25, zpsys='AB', skynoise=0, gain=1
    ... )

Light-curves can then be simulated directly from the model

.. doctest:: python

    >>> # Here we simulate a light-curve with statistical noise
    >>> light_curve = supernova_model.simulate_lc(cadence)

    >>> # Here we simulate a light-curve with a fixed signal to noise ratio
    >>> light_curve_fixed_snr = supernova_model.simulate_lc_fixed_snr(cadence, snr=5)


Module Docs
-----------
"""

from __future__ import annotations

import abc
import warnings
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from typing import *

import joblib
import numpy as np
import pandas as pd
import sncosmo
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from pwv_kpno.defaults import v1_transmission
from pwv_kpno.gps_pwv import GPSReceiver
from pwv_kpno.transmission import calc_pwv_eff
from pytz import utc
from scipy.interpolate import RegularGridInterpolator

from snat_sim.utils.caching import Cache
from . import constants as const
from ._data_paths import data_paths
from .utils import time_series as tsu

Numeric = Union[float, int]
ModelParams = Dict[str, Numeric]
FloatOrArray = TypeVar('FloatOrArray', Numeric, Collection[Numeric], np.ndarray)

# Todo: These were picked ad-hock and are likely too big.
#  They should be set to a reasonable number further along in development
PWV_CACHE_SIZE = 500_000
TRANSMISSION_CACHE_SIZE = 500_00
AIRMASS_CACHE_SIZE = 250_000


@dataclass
class ObservedCadence:
    """The observational sampling of an astronomical light-curve

    The zero-point, zero point system, and gain arguments can be a
    collection of values (one per ``obs_time`` value), or a single value
    to apply at all observation times.

    Args:
        obs_times: Array of observation times for the light-curve
        bands: Array of bands for each observation
        zp: The zero-point or an array of zero-points for each observation
        zpsys: The zero-point system or an array of zero-point systems
        gain: The simulated gain or an array of gain values
    """

    obs_times: Collection[float]
    bands: Collection[str]
    skynoise: FloatOrArray
    zp: FloatOrArray
    zpsys: Union[str, Collection[str]]
    gain: FloatOrArray

    @property
    def skynoise(self) -> np.array:
        return self._skynoise.copy()

    @skynoise.setter
    def skynoise(self, skynoise: FloatOrArray):
        self._skynoise = np.full_like(self.obs_times, skynoise)

    @property
    def zp(self) -> np.array:
        return self._zp.copy()

    @zp.setter
    def zp(self, zp: FloatOrArray):
        self._zp = np.full_like(self.obs_times, zp)

    @property
    def zpsys(self) -> np.array:
        return self._zpsys.copy()

    @zpsys.setter
    def zpsys(self, zpsys: FloatOrArray):
        self._zpsys = np.full_like(self.obs_times, zpsys, dtype='U8')

    @property
    def gain(self) -> np.array:
        return self._gain.copy()

    @gain.setter
    def gain(self, gain: FloatOrArray):
        self._gain = np.full_like(self.obs_times, gain)

    @staticmethod
    def from_plasticc(
            light_curve: Table,
            zp: FloatOrArray = None,
            drop_nondetection: bool = False
    ) -> Tuple[ModelParams, ObservedCadence]:
        """Extract the observational cadence from a PLaSTICC light-curve

        The zero-point, zero point system, and gain arguments can be a
        collection of values (one per phase value), or a single value to
        apply at all obs_times.

        Args:
            light_curve: Astropy table with PLaSTICC light-curve data
            zp: Optionally overwrite the PLaSTICC zero-point with this value(s)
            drop_nondetection: Drop data with PHOTFLAG == 0

        Returns:
            An ``ObservedCadence`` instance
        """

        if drop_nondetection:
            light_curve = light_curve[light_curve['PHOTFLAG'] != 0]

        params = {
            'SNID': light_curve.meta['SNID'],
            'ra': light_curve.meta['RA'],
            'dec': light_curve.meta['DECL'],
            't0': light_curve.meta['SIM_PEAKMJD'],
            'x1': light_curve.meta['SIM_SALT2x1'],
            'c': light_curve.meta['SIM_SALT2c'],
            'z': light_curve.meta['SIM_REDSHIFT_CMB'],
            'x0': light_curve.meta['SIM_SALT2x0']
        }

        return params, ObservedCadence(
            obs_times=light_curve['MJD'],
            bands=['lsst_hardware_' + f.lower().strip() for f in light_curve['FLT']],
            zp=zp or light_curve['ZEROPT'],
            zpsys='AB',
            gain=1,
            skynoise=light_curve['SKY_SIG']
        )

    def to_sncosmo(self) -> Table:
        """Return the observational cadence as an ``astropy.Table``

        The returned table of observations is formatted for use with with
        the ``sncosmo`` package.

        Returns:
            An astropy table representing the observational cadence in ``sncosmo`` format
        """

        observations = Table(
            {
                'time': self.obs_times,
                'band': self.bands,
                'gain': self.gain,
                'skynoise': self.skynoise,
                'zp': self.zp,
                'zpsys': self.zpsys
            },
            dtype=[float, 'U1000', float, float, float, 'U100']
        )

        observations.sort('time')
        return observations

    def __repr__(self) -> str:
        repr_list = self.to_sncosmo().__repr__().split('\n')
        repr_list[0] = super(ObservedCadence, self).__repr__()
        repr_list.pop(2)
        return '\n'.join(repr_list)


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

        self.calc_transmission = Cache('pwv', 'wave', cache_size=TRANSMISSION_CACHE_SIZE)(self.calc_transmission)

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def calc_transmission(self, pwv: float, wave: Optional[np.array] = None) -> pd.Series:
        ...

    # noinspection PyMissingOrEmptyDocstring
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

        self.pwv_model_data = pwv_series.tsu.resample_data_across_year().tsu.periodic_interpolation()
        self.pwv_model_data.index = tsu.datetime_to_sec_in_year(self.pwv_model_data.index)

        self.pwv_los = Cache('time', cache_size=PWV_CACHE_SIZE)(self.pwv_los)

        memory = joblib.Memory(str(data_paths.joblib_path), verbose=0, bytes_limit=AIRMASS_CACHE_SIZE)
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
        supp_data = weather_data.tsu.supplemented_data(year, supp_years)
        return PWVModel(supp_data)

    # noinspection PyMissingOrEmptyDocstring
    @overload
    @staticmethod
    def calc_airmass(
            time: float,
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> float:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    @staticmethod
    def calc_airmass(
            time: Union[np.ndarray, Collection],
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> np.array:
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

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def pwv_zenith(self, time: float, time_format: Optional[str]) -> float:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def pwv_zenith(self, time: Collection[float], time_format: Optional[str]) -> np.array:
        ...

    def pwv_zenith(self, time, time_format='mjd'):
        """Interpolate the PWV at zenith as a function of time

        The ``time_format`` argument can be set to ``None`` when passing datetime
        objects for ``time`` instead of numerical values.

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

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def pwv_los(
            self,
            time: float,
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> float:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def pwv_los(
            self, time: Union[np.ndarray, Collection],
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> np.array:
        ...

    def pwv_los(
            self, time, ra, dec,
            lat=const.vro_latitude,
            lon=const.vro_longitude,
            alt=const.vro_altitude,
            time_format='mjd'
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

    def simulate_lc(self, cadence: ObservedCadence, scatter: bool = True) -> Table:
        """Simulate a SN light-curve

        If ``scatter`` is ``True``, then simulated flux values include an added
        random component drawn from a normal distribution with a standard deviation
        equal to the error of the observation.

        Args:
            cadence: Observational cadence to evaluate the light-curve with
            scatter: Whether to add random noise to the flux values

        Returns:
            The simulated light-curve as an astropy table in the ``sncosmo`` format
        """

        flux = self.bandflux(cadence.bands, cadence.obs_times, zp=cadence.zp, zpsys=cadence.zpsys)
        fluxerr = np.sqrt(cadence.skynoise ** 2 + np.abs(flux) / cadence.gain)

        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        return Table(
            data=[cadence.obs_times, cadence.bands, flux, fluxerr, cadence.zp, cadence.zpsys],
            names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'),
            meta=dict(zip(self.param_names, self.parameters)))

    def simulate_lc_fixed_snr(self, cadence: ObservedCadence, snr: float = .05) -> Table:
        """Simulate a SN light-curve with a fixed SNR for the given cadence

        Args:
            cadence: Observational cadence to evaluate the light-curve with
            snr: Signal to noise ratio

        Returns:
            The simulated light-curve as an astropy table in the ``sncosmo`` format
        """

        obs = cadence.to_sncosmo()
        light_curve = obs[['time', 'band', 'zp', 'zpsys']]
        light_curve['flux'] = self.bandflux(obs['band'], obs['time'], obs['zp'], obs['zpsys'])
        light_curve['fluxerr'] = light_curve['flux'] / snr
        light_curve.meta = dict(zip(self.param_names, self.parameters))
        return light_curve


###############################################################################
# Propagation effects
###############################################################################


class VariablePropagationEffect(sncosmo.PropagationEffect):
    """Base class for propagation effects that vary with time

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

    def _apply_propagation(self, flux: np.ndarray, transmission: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """Apply an atmospheric transmission to flux values

        Args:
            flux: Array of flux values
            transmission: Array of sampled transmission values
        """

        if isinstance(transmission, pd.DataFrame):  # PWV is a vector and transmission is a DataFrame
            return flux * transmission.values.T

        else:  # Assume PWV is scalar and transmission is Series-like
            if np.ndim(flux) == 1:
                return flux * transmission

            if np.ndim(flux) == 2:
                return flux * np.atleast_2d(transmission)

        raise NotImplementedError('Could not identify how to match dimensions of atm. model to source flux.')

    def assumed_pwv(self, time: FloatOrArray) -> FloatOrArray:
        """The PWV concentration used by the propagation effect at a given time

        Args:
            time): Time to get the PWV concentration for

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
        return self._apply_propagation(flux, transmission)


class SeasonalPWVTrans(VariablePWVTrans):
    """Atmospheric propagation effect for a fixed PWV concentration per-season"""

    def assumed_pwv(self, time: FloatOrArray) -> FloatOrArray:
        """The PWV concentration used by the propagation effect at a given time

        Args:
            time: Time to get the PWV concentration for

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
            wave: An array of wavelength values
            flux: An array of flux values
            time: Array of time values to determine PWV for

        Returns:
            An array of flux values after suffering from PWV absorption
        """

        pwv = self.assumed_pwv(time)
        transmission = self._transmission_model.calc_transmission(pwv, np.atleast_1d(wave))
        return self._apply_propagation(flux, transmission)
