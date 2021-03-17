"""Modeling functionality for the spectro-photometric time-sampling of
supernovae.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from typing import *
from typing import Optional, Union

import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table

from .pwv import VariablePropagationEffect
from .. import types
from ..utils import cov_utils as cutils


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
    skynoise: types.FloatColl
    zp: types.FloatColl
    zpsys: Union[str, Collection[str]]
    gain: types.FloatColl

    def __eq__(self, other: ObservedCadence) -> bool:
        attr_list = ['obs_times', 'bands', 'skynoise', 'zp', 'zpsys', 'gain']
        return np.all(np.equal(getattr(self, attr), getattr(other, attr)) for attr in attr_list)

    @property
    def skynoise(self) -> np.array:
        return self._skynoise.copy()

    @skynoise.setter
    def skynoise(self, skynoise: types.FloatColl):
        self._skynoise = np.full_like(self.obs_times, skynoise)

    @property
    def zp(self) -> np.array:
        return self._zp.copy()

    @zp.setter
    def zp(self, zp: types.FloatColl):
        self._zp = np.full_like(self.obs_times, zp)

    @property
    def zpsys(self) -> np.array:
        return self._zpsys.copy()

    @zpsys.setter
    def zpsys(self, zpsys: types.FloatColl):
        self._zpsys = np.full_like(self.obs_times, zpsys, dtype='U8')

    @property
    def gain(self) -> np.array:
        return self._gain.copy()

    @gain.setter
    def gain(self, gain: types.FloatColl):
        self._gain = np.full_like(self.obs_times, gain)

    @staticmethod
    def from_plasticc(
            light_curve: Table,
            zp: types.FloatColl = None,
            drop_nondetection: bool = False
    ) -> Tuple[types.NumericalParams, ObservedCadence]:
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
            'SNID': light_curve.meta['SNID'].strip(),
            'ra': light_curve.meta['RA'],
            'dec': light_curve.meta['DECL'],
            't0': light_curve.meta['SIM_PEAKMJD'],
            'x1': light_curve.meta['SIM_SALT2x1'],
            'c': light_curve.meta['SIM_SALT2c'],
            'z': light_curve.meta['SIM_REDSHIFT_CMB'],
            'x0': light_curve.meta['SIM_SALT2x0']
        }

        return cast(types.NumericalParams, params), ObservedCadence(
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

    def __repr__(self) -> str:  # pragma: no cover
        repr_list = self.to_sncosmo().__repr__().split('\n')
        repr_list[0] = super(ObservedCadence, self).__repr__()
        repr_list.pop(2)
        return '\n'.join(repr_list)


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

    def simulate_lc(self, cadence: ObservedCadence, scatter: bool = True, fixed_snr: Optional[float] = None) -> Table:
        """Simulate a SN light-curve

        If ``scatter`` is ``True``, then simulated flux values include an added
        random component drawn from a normal distribution with a standard deviation
        equal to the error of the observation.

        Args:
            cadence: Observational cadence to evaluate the light-curve with
            scatter: Whether to add random noise to the flux values
            fixed_snr: Optionally simulate the light-curve using a fixed signal to noise ratio

        Returns:
            The simulated light-curve as an astropy table in the ``sncosmo`` format
        """

        flux = self.bandflux(cadence.bands, cadence.obs_times, zp=cadence.zp, zpsys=cadence.zpsys)

        if fixed_snr:
            fluxerr = flux / fixed_snr

        else:
            fluxerr = np.sqrt(cadence.skynoise ** 2 + np.abs(flux) / cadence.gain)

        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        return Table(
            data=[cadence.obs_times, cadence.bands, flux, fluxerr, cadence.zp, cadence.zpsys],
            names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'),
            meta=dict(zip(self.param_names, self.parameters)))

    def fit_lc(
            self,
            data: Table = None,
            vparam_names: List = tuple(),
            bounds: Dict[str: Tuple[types.Numeric, types.Numeric]] = None,
            method: str = 'minuit',
            guess_amplitude: bool = True,
            guess_t0: bool = True,
            guess_z: bool = True,
            minsnr: float = 5.0,
            modelcov: bool = False,
            maxcall: int = 10000,
            phase_range: List[types.Numeric] = None,
            wave_range: List[types.Numeric] = None
    ) -> Tuple[SNFitResult, SNModel]:
        """Fit model parameters to photometric data via a chi-squared minimization

        Fitting behavior:
          - Parameters of the parent instance are not modified during the fit
          - If ``modelcov`` is enabled, the fit is performed multiple times until convergence.
          - The ``t0`` parameter has a default fitting boundary such that the latest
            phase of the model lines up with the earliest data point and the earliest
            phase of the model lines up with the latest data point.

        Args:
            data: Table of photometric data.
            vparam_names: Model parameters to vary in the fit.
            bounds: Bounded range for each parameter. Keys should be parameter names, values are tuples.
            guess_amplitude: Whether or not to guess the amplitude from the data.
            guess_t0: Whether or not to guess t0. Only has an effect when fitting t0.
            guess_z: Whether or not to guess z (redshift). Only has an effect when fitting redshift.
            minsnr: When guessing amplitude and t0, only use data with signal-to-noise ratio greater than this value.
            method: Minimization method to use. Currently there is only one choice.
            modelcov: Include model covariance when calculating chisq. Default is False.
            maxcall: Maximum number of chi-square iterations to evaluate when fitting.
            phase_range: If given, discard data outside this range of phases.
            wave_range: If given, discard data with bandpass effective wavelengths outside this range.

        Returns:
            The fit result and a copy of the model set to the fitted parameters
        """

        try:
            fit_func = {'iminuit': sncosmo.fit_lc, 'emcee': sncosmo.mcmc_lc}[method]

        except KeyError:
            raise ValueError(f'Invalid fitting method: {method}')

        result, fitted_model = fit_func(
            data=data,
            model=deepcopy(self),
            vparam_names=vparam_names,
            bounds=bounds, method=method,
            guess_amplitude=guess_amplitude,
            guess_t0=guess_t0,
            guess_z=guess_z,
            minsnr=minsnr,
            modelcov=modelcov,
            maxcall=maxcall,
            phase_range=phase_range,
            wave_range=wave_range,
            warn=False
        )

        return SNFitResult(result), SNModel(fitted_model)


class SNFitResult(sncosmo.utils.Result):

    @property
    def param_names(self) -> List[str]:
        return copy(self['param_names'])

    @property
    def parameters(self) -> pd.Series:
        """The model parameters"""

        return pd.Series(self['parameters'], index=self.param_names)

    @property
    def vparam_names(self) -> List[str]:
        return copy(self['vparam_names'])

    @property
    def vparameters(self) -> pd.Series:
        """The values of the varied parameters"""

        vparameters = [self['parameters'][self['param_names'].index(v)] for v in self.vparam_names]
        return pd.Series(vparameters, index=self.vparam_names)

    @property
    def covariance(self) -> pd.DataFrame:
        """The covariance matrix"""

        return cutils.covariance(self['covariance'], paramNames=self.vparam_names)

    def salt_covariance_linear(self, x0Truth: float = None) -> pd.DataFrame:
        """The covariance matrix of apparent magnitude and salt2 parameters"""

        x0 = self.parameters.loc['x0'] if x0Truth is None else x0Truth

        factor = - 2.5 / np.log(10)
        # drop other parameters like t0
        cov = self.covariance.copy()
        cov = cutils.subcovariance(covariance=cov, paramList=['x0', 'x1', 'c'], array=False)
        covariance = cutils.log_covariance(cov, paramName='x0', paramValue=x0, factor=factor)

        covariance.rename(columns={'x0': 'mB'}, inplace=True)
        covariance['name'] = covariance.columns
        covariance.set_index('name', inplace=True)
        covariance.index.name = None

        return covariance

    def mu_variance_linear(self, alpha: float = 0.14, beta: float = 3.1) -> float:
        """Calculate the variance in distance modulus

        Determined using the covariance matrix of apparent magnitude and
        salt2 parameters (See the ``salt_covariance_linear`` method).

        Args:
            alpha: The stretch correction factor
            beta: The color excess correction factor

        Returns:
            The variance in mu
        """

        arr = np.array([1.0, alpha, -beta])
        _cov = self.salt_covariance_linear()
        sc = cutils.subcovariance(_cov, paramList=['mB', 'x1', 'c'], array=True)
        return cutils.expAVsquare(sc, arr)

    def __repr__(self):
        # Extremely similar to the base representation of the parent class but
        # cleaned up so values are displayed in neat rows / columns

        four_spaces = '    '
        with np.printoptions(precision=3):
            chisq_str = str(np.array([self.chisq]))[1:-1]
            params_str = str(self.parameters.values)
            covariance_str = four_spaces + str(self.covariance.values).replace('\n', f'\n{four_spaces}')
            errors_str = str(np.array(list(self.errors.values())))

        return (
                f"     success: {self.success}\n"
                f"     message: {self.message}\n"
                f"       ncall: {self.ncall}\n"
                f"        nfit: {self.nfit}\n"
                f"       chisq: {chisq_str}\n"
                f"        ndof: {self.ndof}\n"
                f" param_names: {self.param_names}\n"
                f"  parameters: {params_str}\n"
                f"vparam_names: {self.vparam_names}\n"
                f"      errors: {errors_str}\n"
                f"  covariance:\n"
                + covariance_str
        )
