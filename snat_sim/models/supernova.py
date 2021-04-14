"""Modeling functionality for the spectro-photometric time-sampling of
supernovae.
"""

from __future__ import annotations

from copy import copy, deepcopy
from typing import *
from typing import Optional

import numpy as np
import pandas as pd
import sncosmo

from .light_curve import LightCurve, ObservedCadence
from .pwv import VariablePropagationEffect
from .. import types


class SNModel(sncosmo.Model):
    """An observer-frame supernova model composed of a Source and zero or more effects"""

    @staticmethod
    def from_sncosmo(model: sncosmo.Model) -> SNModel:
        """Create an `SNModel`` instance from a ``sncosmo.Model`` instance

        Args:
            model: The sncosmo model to build from

        Returns:
            An ``SNModel`` object
        """

        new_model = SNModel(
            model.source,
            effects=model.effects,
            effect_names=model.effect_names,
            effect_frames=model._effect_frames)

        new_model.update({p: model[p] for p in new_model.param_names})
        return new_model

    # Same as parent except allows duck-typing of ``effect`` arg
    def _add_effect_partial(self, effect, name, frame) -> None:
        """Like ``add effect``, but don't sync parameter arrays"""

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

        # Note that below we multiply by the scale factor to conserve bolometric luminosity.
        # noinspection PyProtectedMember
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

    def simulate_lc(
            self, cadence: ObservedCadence, scatter: bool = True, fixed_snr: Optional[float] = None
    ) -> LightCurve:
        """Simulate a SN light-curve

        If ``scatter`` is ``True``, then simulated flux values include an added
        random component drawn from a normal distribution with a standard deviation
        equal to the error of the observation.

        Args:
            cadence: Observational cadence to evaluate the light-curve with
            scatter: Whether to add random noise to the flux values
            fixed_snr: Optionally simulate the light-curve using a fixed signal to noise ratio

        Returns:
            The simulated light-curve
        """

        flux = self.bandflux(cadence.bands, cadence.obs_times, zp=cadence.zp, zpsys=cadence.zpsys)

        if fixed_snr:
            fluxerr = flux / fixed_snr

        else:
            fluxerr = np.sqrt(cadence.skynoise ** 2 + np.abs(flux) / cadence.gain)

        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        return LightCurve(
            time=cadence.obs_times,
            band=cadence.bands,
            flux=flux,
            fluxerr=fluxerr,
            zp=cadence.zp,
            zpsys=cadence.zpsys
        )

    def fit_lc(
            self,
            data: LightCurve = None,
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
            method: Minimization method to use. Either "minuit" or "emcee"
            modelcov: Include model covariance when calculating chisq. Default is False.
            maxcall: Maximum number of chi-square iterations to evaluate when fitting.
            phase_range: If given, discard data outside this range of phases.
            wave_range: If given, discard data with bandpass effective wavelengths outside this range.

        Returns:
            The fit result and a copy of the model set to the fitted parameters
        """

        try:
            fit_func = {'minuit': sncosmo.fit_lc, 'emcee': sncosmo.mcmc_lc}[method]

        except KeyError:
            raise ValueError(f'Invalid fitting method: {method}')

        result, fitted_model = fit_func(
            data=data.to_astropy(),
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

        return SNFitResult(result), SNModel.from_sncosmo(fitted_model)


class SNFitResult(sncosmo.utils.Result):
    """Represents results from a ``SNModel`` being fit to a ``LightCurve``"""

    def __eq__(self, other):

        if isinstance(other, self.__class__) and (self.keys() == other.keys()):
            for key, val in self.items():
                if not np.array_equal(val, other[key]):
                    return False

            return True
        return False

    @property
    def param_names(self) -> List[str]:
        """The names of the model parameters"""

        return copy(self['param_names'])

    @property
    def parameters(self) -> pd.Series:
        """The model parameters"""

        return pd.Series(self['parameters'], index=self.param_names)

    @property
    def vparam_names(self) -> List[str]:
        """List of parameter names varied in the fit"""

        return copy(self['vparam_names'])

    @property
    def vparameters(self) -> pd.Series:
        """The values of the varied parameters"""

        vparameters = [self['parameters'][self['param_names'].index(v)] for v in self.vparam_names]
        return pd.Series(vparameters, index=self.vparam_names)

    @property
    def covariance(self) -> Optional[pd.DataFrame]:
        """The covariance matrix"""

        if self['covariance'] is None:
            return None

        return pd.DataFrame.cov_utils.from_array(self['covariance'], paramNames=self.vparam_names)

    def salt_covariance_linear(self, x0_truth: float = None) -> pd.DataFrame:
        """The covariance matrix of apparent magnitude and salt2 parameters

        Will raise an error if the `x0`, `x1` and `c` parameters are not
        varied in the fit.

        Args:
            x0_truth: Optionally assert an alternative x0 value

        Returns:
            The covariance matrix asd a pandas ``DataFrame``
        """

        if not self.success:
            raise RuntimeError('Cannot calculate variance for a failed fit.')

        x0 = self.parameters.loc['x0'] if x0_truth is None else x0_truth

        factor = - 2.5 / np.log(10)
        # drop other parameters like t0
        cov = self.covariance.copy()
        cov = cov.cov_utils.subcovariance(paramList=['x0', 'x1', 'c'])
        covariance = cov.cov_utils.log_covariance(paramName='x0', paramValue=x0, factor=factor)

        covariance.rename(columns={'x0': 'mB'}, inplace=True)
        covariance['name'] = covariance.columns
        covariance.set_index('name', inplace=True)
        covariance.index.name = None

        return covariance

    def mu_variance_linear(self, alpha: float, beta: float) -> float:
        """Calculate the variance in distance modulus

        Determined using the covariance matrix of apparent magnitude and
        salt2 parameters (See the ``salt_covariance_linear`` method).

        Args:
            alpha: The stretch correction factor
            beta: The color excess correction factor

        Returns:
            The variance in mu
        """

        if not self.success:
            raise RuntimeError('Cannot calculate variance for a failed fit.')

        arr = np.array([1.0, alpha, -beta])
        _cov = self.salt_covariance_linear()
        sc = _cov.cov_utils.subcovariance(paramList=['mB', 'x1', 'c'])
        return sc.cov_utils.expAVsquare(arr)

    def __repr__(self):
        # Extremely similar to the base representation of the parent class but
        # cleaned up so values are displayed in neat rows / columns

        with np.printoptions(precision=3):
            chisq_str = str(np.array([self.chisq]))[1:-1]
            params_str = str(self.parameters.values)
            errors_str = str(np.array(list(self.errors.values())))
            if self.covariance is not None:
                four_spaces = '\n    '
                covariance_str = four_spaces + str(self.covariance.values).replace('\n', f'{four_spaces}')

            else:
                covariance_str = ' None'

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
            f"  covariance:{covariance_str}"
        )
