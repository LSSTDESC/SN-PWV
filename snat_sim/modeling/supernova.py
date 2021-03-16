from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import *

import numpy as np
import sncosmo
from astropy.table import Table

from .pwv import VariablePropagationEffect
from .. import types


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
