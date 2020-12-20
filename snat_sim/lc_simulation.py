"""The ``lc_simulation`` module realizes light-curves for a given temporal sampling
(the "cadence") and supernova model.

Usage Example
-------------

To simulate a light-curve, you must first establish the desired light-curve
cadence (i.e., how the light-curve should be sampled in time). Using your
supernova model of choice, light-curves can then be simulated with and
without statistical noise. Both approaches are demonstrated below using the
``salt2`` supernova model.

.. doctest:: python

    >>> from snat_sim import lc_simulation, models

    >>> # We start by defining the observational cadence
    >>> cadence = ObservedCadence(
    ...     obs_times=[-1, 0, 1],
    ...     bands=['sdssr', 'sdssr', 'sdssr'],
    ...     zp=25, zpsys='AB', skynoise=0, gain=1
    ... )

    >>> # Next we choose the supernova model and desired model parameters
    >>> sn_model = models.SNModel('salt2')
    >>> sn_parameters = dict(z=.01, x1=.5, c=-.1)


    >>> # Finally, we simulate a light-curve
    >>> simulator = LCSimulator(sn_model, cadence)
    >>> with_noise = simulator.simulate_lc(sn_parameters)
    >>> without_noise = simulator.simulate_lc_fixed_snr(sn_parameters, snr=6)

Instances of the ``LCSimulator`` class will default to using parameter values
set in the suypernova model at instantiation. These can be changed as follows:

.. doctest:: python

    >>> sn_model = models.SNModel('salt2')
    >>> sn_model.set(z=.01, x1=.5, c=-.1)
    >>> simulator = LCSimulator(sn_model, cadence)
    >>> print(simulator.default_parameters)

    >>> # Finally, we simulate a light-curve
    >>> simulator.model.set(c=.5)
    >>> print(simulator.default_parameters)

Module Docs
-----------
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import *

import numpy as np
from astropy.cosmology.core import Cosmology
from astropy.table import Table

from . import constants as const
from .models import SNModel

Numeric = Union[float, int]


class ObservedCadence:
    """Represents the temporal sampling of an observed supernova light-curve"""

    def __init__(
            self,
            obs_times: Collection[float],
            bands: Collection[str],
            skynoise: Union[Numeric, Collection[Numeric]],
            zp: Union[Numeric, Collection[Numeric]],
            zpsys: Union[str, Collection[str]],
            gain: Union[Numeric, Collection[Numeric]]
    ) -> None:
        """The observational sampling of an astronomical light-curve

        The zero-point, zero point system, and gain arguments can be a
        collection of values (one per phase value), or a single value to 
        apply at all phases. 

        Args:
            obs_times: Array of observation times for the light-curve
            bands: Array of bands for each observation
            zp: The zero-point or an array of zero-points for each observation
            zpsys: The zero-point system or an array of zero-point systems
            gain: The simulated gain or an array of gain values
        """

        self.obs_times = obs_times
        self.bands = bands
        self.skynoise = np.full_like(self.obs_times, skynoise)
        self.zp = np.full_like(self.obs_times, zp, dtype=float)
        self.zpsys = np.full_like(self.obs_times, zpsys, dtype='U10')
        self.gain = np.full_like(self.obs_times, gain)

    @staticmethod
    def from_plasticc(
            light_curve: Table,
            zp: Union[Numeric, Collection[Numeric]] = None,
            zpsys: Union[str, Collection[str]] = 'AB',
            gain: Union[Numeric, Collection[Numeric]] = 1,
            drop_nondetection: bool = False
    ) -> ObservedCadence:
        """Extract the observational cadence from a PLaSTICC light-curve
        
        The zero-point, zero point system, and gain arguments can be a
        collection of values (one per phase value), or a single value to 
        apply at all obs_times.

        Args:
            light_curve: Astropy table with PLaSTICC light-curve data
            zp: Optionally overwrite the PLaSTICC zero-point with this value
            zpsys: The zero point system
            gain: The gain value of each observation
            drop_nondetection: Drop data with PHOTFLAG == 0

        Returns:
            An ``ObservedCadence`` instance
        """

        if drop_nondetection:
            light_curve = light_curve[light_curve['PHOTFLAG'] != 0]

        return ObservedCadence(
            obs_times=light_curve['MJD'],
            bands=['lsst_hardware_' + f.lower().strip() for f in light_curve['FLT']],
            zp=zp or light_curve['ZEROPT'],
            zpsys=zpsys,
            gain=gain,
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


@dataclass
class LCSimulator:
    """Handles the simulation of SNe light-curves according to an observational cadence and supernova model"""

    model: SNModel
    cadence: ObservedCadence

    def calc_x0_for_z(
            self,
            z: float,
            cosmo: Cosmology = const.betoule_cosmo,
            abs_mag: float = const.betoule_abs_mb,
            band: str = 'standard::b',
            magsys: str = 'AB',
            **params
    ) -> float:
        """Determine x0 for a given redshift using the current simulation model

        Args:
             z: Redshift to determine x0 for
             cosmo: Cosmology to use when determining x0
             abs_mag: Absolute peak magnitude of the SNe Ia
             band: Band to set absolute magnitude in
             magsys: Magnitude system to set absolute magnitude in
             Any other params to set for the provided `source`

        Returns:
            The x0 parameter for the given source and redshift
        """

        model = copy(self.model)
        model.set(z=z, **params)
        model.set_source_peakabsmag(abs_mag, band, magsys, cosmo=cosmo)
        return model['x0']

    @property
    def default_parameters(self) -> Dict[str, Numeric]:
        """Default model parameters used when simulating light-curves"""

        return dict(zip(self.model.param_names, self.model.parameters))

    def reduce_model_params(self, params: Dict[str:Numeric]) -> Dict[str, Numeric]:
        """Return the subset of model parameters belonging to the simulation model

        Args:
            params: Parameter values representing a superset of the simulation model parameters

        Returns:
            Just the parameters belonging to the simulation model
        """

        return {p: v for p, v in params.items() if p in self.model.param_names}

    def simulate_lc_fixed_snr(self, params: Dict[str, Numeric] = None, snr: float = .05) -> Table:
        """Simulate a SN light-curve with a fixed SNR

        Unless otherwise specified, the scale factor parameter ``x0`` is
        automatically set according to the redshift.

        Args:
            params: Values for any model parameters
            snr: Signal to noise ratio

        Returns:
            An astropy table representing a light-curve in the ``sncosmo`` format
        """

        model = copy(self.model)
        model.update(self.reduce_model_params(params))
        if 'x0' not in params:
            model['x0'] = self.calc_x0_for_z(model['z'])

        obs_table = self.cadence.to_sncosmo()
        light_curve = obs_table[['time', 'band', 'zp', 'zpsys']]
        light_curve['flux'] = model.bandflux(obs_table['band'], obs_table['time'], obs_table['zp'], obs_table['zpsys'])
        light_curve['fluxerr'] = light_curve['flux'] / snr
        light_curve.meta = dict(zip(model.param_names, model.parameters))
        return light_curve

    def simulate_lc(self, params: Dict[str, Numeric] = None, scatter: bool = True) -> Table:
        """Simulate a SN light-curve

        If ``scatter`` is ``True``, then simulated flux values include an added
        random component drawn from a normal distribution with a standard deviation
        equal to the error of the observation.

        Args:
            params: Parameters to feed to the model for realizing the light-curve
            scatter: Whether to add random noise to the flux values

        Returns:
            An astropy table representing a light-curve in the ``sncosmo`` format
        """

        # Update simulation model with params
        params = params or dict()
        model = copy(self.model)
        model.update(self.reduce_model_params(params))

        flux = model.bandflux(
            self.cadence.bands, self.cadence.obs_times, zp=self.cadence.zp, zpsys=self.cadence.zpsys)

        fluxerr = np.sqrt(self.cadence.skynoise ** 2 + np.abs(flux) / self.cadence.gain)

        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        return Table(
            data=[self.cadence.obs_times, self.cadence.bands, flux, fluxerr, self.cadence.zp, self.cadence.zpsys],
            names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'), meta=params)


def duplicate_plasticc_lc(
        light_curve: Table,
        model: SNModel,
        scatter: bool = True,
        cosmo: Optional[Cosmology] = const.betoule_cosmo
) -> Table:
    """Simulate a light-curve with sncosmo that matches the cadence of a PLaSTICC light-curve

    Args:
        light_curve: Astropy table with PLaSTICC light-curve data
        model: The supernova model to use in the simulation
        scatter: Add random noise to the flux values
        cosmo: Optionally rescale the ``x0`` parameter according to the given cosmology

    Returns:
        Astropy table with data for the simulated light-curve
    """

    cadence = ObservedCadence.from_plasticc(light_curve)
    simulator = LCSimulator(model, cadence)

    use_redshift = 'SIM_REDSHIFT_CMB'
    if cosmo is None:
        x0 = light_curve.meta['SIM_SALT2x0']

    else:
        x0 = simulator.calc_x0_for_z(light_curve.meta[use_redshift], cosmo=cosmo)

    # Params double as simulation parameters and meta-data
    params = {
        'SNID': light_curve.meta['SNID'],
        'ra': light_curve.meta['RA'],
        'dec': light_curve.meta['DECL'],
        't0': light_curve.meta['SIM_PEAKMJD'],
        'x1': light_curve.meta['SIM_SALT2x1'],
        'c': light_curve.meta['SIM_SALT2c'],
        'z': light_curve.meta[use_redshift],
        'x0': x0
    }

    # Simulate the light-curve
    return simulator.simulate_lc(params, scatter=scatter)
