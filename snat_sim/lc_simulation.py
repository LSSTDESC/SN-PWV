"""The ``lc_simulation`` module realizes light-curves for a given temporal sampling
(the "cadence") and supernova model.

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
    >>> cadence = lc_simulation.ObservedCadence(bands=band_passes)

    >>> # Evaluate the model at a fixed SNR
    >>> light_curve = cadence.simulate_lc_fixed_snr(sn_model, snr=5)

    >>> # Or, evaluate using statistical uncertainties determined from the gain / skynoise
    >>> light_curve = cadence.simulate_lc(sn_model)


Module Docs
-----------
"""

from __future__ import annotations

from copy import copy
from typing import *

import numpy as np
import sncosmo
from astropy.cosmology.core import Cosmology
from astropy.table import Table

from . import constants as const
from .models import SNModel

Numeric = Union[float, int]


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


def duplicate_plasticc_sncosmo(
        light_curve: Table,
        model: SNModel,
        zp: Numeric = None,
        gain: Numeric = 1,
        skynoise: Numeric = None,
        scatter: bool = True,
        cosmo: Optional[Cosmology] = const.betoule_cosmo
) -> Table:
    """Simulate a light-curve with sncosmo that matches the cadence of a PLaSTICC light-curve

    Args:
        light_curve: Astropy table with PLaSTICC light-curve data
        model: SNModel to use when simulating light-curve flux
        zp: Optionally overwrite the PLaSTICC zero-point with this value
        gain: Gain to use during simulation
        skynoise:  Optionally overwrite the PLaSTICC skynoise with this value
        scatter: Add random noise to the flux values
        cosmo: Optionally rescale the ``x0`` parameter according to the given cosmology

    Returns:
        Astropy table with data for the simulated light-curve
    """

    use_redshift = 'SIM_REDSHIFT_CMB'
    if cosmo is None:
        x0 = light_curve.meta['SIM_SALT2x0']

    else:
        x0 = calc_x0_for_z(light_curve.meta[use_redshift], 'salt2', cosmo=cosmo)

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
    zp = zp if zp is not None else light_curve['ZEROPT']
    skynoise = skynoise if skynoise is not None else light_curve['SKY_SIG']
    observations = ObservedCadence.from_plasticc(light_curve, zp=zp, gain=gain, skynoise=skynoise)
    return observations.simulate_lc(model, params, scatter=scatter)


class ObservedCadence:
    """Represents the temporal sampling of an observed supernova light-curve"""

    def __init__(
            self,
            phases: Collection[float] = range(-20, 51),
            bands: Collection[str] = ('decam_g', 'decam_r', 'decam_i', 'decam_z', 'decam_y'),
            zp: Union[int, float] = 25,
            zpsys: str = 'AB',
            gain: int = 100
    ) -> None:
        """

        ``phases`` are specified in units of phase by default, but can be chosen
        to reflect any time convention.

        Args:
            phases: Array of phase values to include
            bands: Array of bands to include
            zp: The zero point
            zpsys: The zero point system
            gain: The simulated gain
        """

        self.phase_arr = np.concatenate([phases for _ in bands])
        self.band_arr = np.concatenate([np.full_like(phases, b, dtype='U1000') for b in bands])
        self.gain_arr = np.full_like(self.phase_arr, gain)
        self.skynoise_arr = np.zeros_like(self.phase_arr)
        self.zp_arr = np.full_like(self.phase_arr, zp, dtype=float)
        self.zp_sys_arr = np.full_like(self.phase_arr, zpsys, dtype='U10')

    @staticmethod
    def from_plasticc(
            light_curve: Table,
            zp: Numeric = 25,
            gain: Numeric = 1,
            skynoise: Numeric = 0,
            drop_nondetection: bool = False
    ) -> ObservedCadence:
        """Extract the observational cadence from a PLaSTICC light-curve

        Returned table is formatted for use with ``sncosmo.realize_lcs``.

        Args:
            light_curve      (Table): Astropy table with PLaSTICC light-curve data
            zp        (float, array): Overwrite the PLaSTICC zero-point with this value
            gain             (float): Gain to use during simulation
            skynoise    (int, array): Simulated skynoise in counts
            drop_nondetection (bool): Drop data with PHOTFLAG == 0

        Returns:
            An astropy table with cadence data for the input light-curve
        """

        if drop_nondetection:
            light_curve = light_curve[light_curve['PHOTFLAG'] != 0]

        observations = Table({
            'time': light_curve['MJD'],
            'band': ['lsst_hardware_' + f.lower().strip() for f in light_curve['FLT']],
        })

        observations['zp'] = zp
        observations['zpsys'] = 'ab'
        observations['gain'] = gain
        observations['skynoise'] = skynoise
        return observations  Todo: Fix return type

    def to_sncosmo(self) -> Table:
        """Return the observational cadence as an ``astropy.Table`` formatted for use with ``sncosmo``

        Returns:
            An astropy table
        """

        observations = Table(
            {
                'time': self.phase_arr,
                'band': self.band_arr,
                'gain': self.gain_arr,
                'skynoise': self.skynoise_arr,
                'zp': self.zp_arr,
                'zpsys': self.zp_sys_arr
            },
            dtype=[float, 'U1000', float, float, float, 'U100']
        )

        observations.sort('time')
        return observations

    # Todo: The default gain value should match create_observations_table
    def simulate_lc_fixed_snr(self, model: SNModel, snr: float = .05, **params) -> Table:
        """Simulate a SN light-curve with a fixed SNR given a set of observations

        The ``obs`` table is expected to have columns for 'time', 'band', 'zp',
        and 'zpsys'.

        Args:
            model: Supernova model to evaluate
            snr: Signal to noise ratio
            **params: Values for any model parameters

        Returns:
            An astropy table formatted for use with ``sncosmo``
        """

        model = copy(model)
        model.update(params)

        # Set default x0 value according to assumed cosmology and the model redshift
        x0 = params.get('x0', calc_x0_for_z(model['z'], model.source))
        model.set(x0=x0)

        obs_table = self.to_sncosmo()
        light_curve = obs_table[['time', 'band', 'zp', 'zpsys']]
        light_curve['flux'] = model.bandflux(obs_table['band'], obs_table['time'], obs_table['zp'], obs_table['zpsys'])
        light_curve['fluxerr'] = light_curve['flux'] / snr
        light_curve.meta = dict(zip(model.param_names, model.parameters))
        return light_curve

    def simulate_lc(self, model: SNModel, params: Dict[str, float] = None, scatter: bool = True) -> Table:
        """Simulate a SN light-curve given a set of observations

        If ``scatter`` is ``True``, then simulated flux values include an added
        random number drawn from a Normal Distribution with a standard deviation
        equal to the error of the observation.

        Args:
            model: The model to use in the simulations
            params: parameters to feed to the model for realizing the light-curve
            scatter: Add random noise to the flux values

        Returns:
            An astropy table formatted for use with ``sncosmo``
        """

        if params is None:
            params = dict()

        model = copy(model)
        for p in model.param_names:
            model[p] = params.get(p, model[p])

        flux = model.bandflux(self.band_arr, self.phase_arr, zp=self.zp_arr, zpsys=self.zp_sys_arr)
        fluxerr = np.sqrt(self.skynoise_arr ** 2 + np.abs(flux) / self.gain_arr)

        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        return Table(
            data=[self.phase_arr, self.band_arr, flux, fluxerr, self.zp_arr, self.zp_sys_arr],
            names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'), meta=params)
