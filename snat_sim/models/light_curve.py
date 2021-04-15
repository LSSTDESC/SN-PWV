"""Abstract representations of astronomical time-series data"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import *
from typing import Optional, Union

import numpy as np
import pandas as pd
import sncosmo.photdata
from astropy.table import Table

from .. import types

SNCOSMO_ALIASES = dict()
for column_name, alias_set in sncosmo.photdata.PHOTDATA_ALIASES.items():
    for alias in alias_set:
        SNCOSMO_ALIASES[alias] = column_name


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
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


class LightCurve:
    """Abstract representation of an astronomical light-curve."""

    def __init__(
            self,
            time: Collection[float],
            band: Collection[str],
            flux: Collection[float],
            fluxerr: Collection[float],
            zp: Collection[float],
            zpsys: Collection[str],
            phot_flag: Optional[Collection[int]] = None
    ) -> None:
        """An astronomical light-curve

        Args:
            time: Time values for each observation in a numerical format (e.g. JD or MJD)
            band: The band each observation was performed in
            flux: The flux of each observation
            fluxerr: The error in each flux value
            zp: The zero-point of each observation
            zpsys: The zero-point system of each observation
            phot_flag: Optional flag for each photometric observation
        """

        phot_flag = np.full_like(time, 0) if phot_flag is None else phot_flag
        self.time = pd.Index(time, name='time')
        self.band = pd.Series(band, name='band', index=self.time)
        self.flux = pd.Series(flux, name='flux', index=self.time)
        self.fluxerr = pd.Series(fluxerr, name='fluxerr', index=self.time)
        self.zp = pd.Series(zp, name='zp', index=self.time)
        self.zpsys = pd.Series(zpsys, name='zpsys', index=self.time)
        self.phot_flag = pd.Series(phot_flag, name='phot_flag', index=self.time)

    @staticmethod
    def from_sncosmo(data: Table) -> LightCurve:
        """Create a ``LightCurve`` instance from an astropy table in the SNCosmo format

        Args:
            data: A table in the sncosmo format

        Returns:
            A ``LightCurve`` instance
        """

        # The sncosmo data format uses flexible column names
        # E.g., 'time', 'date', 'jd', 'mjd', 'mjdobs', and 'mjd_obs' are all equivalent
        # Here we map those column names onto the kwarg names for the parent class
        return LightCurve(**{SNCOSMO_ALIASES.get(col, col): data[col] for col in data.colnames})

    def to_astropy(self) -> Table:
        """Return the light-curve data as an astropy ``Table`` formatted for use with sncosmo

        Returns:
            An ``Table`` instance formatted for use with sncosmo
        """

        return Table.from_pandas(self.to_pandas().reset_index())

    def to_pandas(self) -> pd.DataFrame:
        """Return the light-curve data as a pandas ``DataFrame``

        Returns:
            A ``DataFrame`` instance with the light-curve data
        """

        return pd.DataFrame(dict(
            band=self.band,
            flux=self.flux,
            fluxerr=self.fluxerr,
            zp=self.zp,
            zpsys=self.zpsys,
            phot_flag=self.phot_flag
        ), index=self.time)

    def __len__(self) -> int:
        """The number of observations in the light-curve"""

        return len(self.band)

    def copy(self) -> LightCurve:
        """Return a copy of the instance"""

        return copy(self)

    def __eq__(self, other: LightCurve):
        """Check whether both objects have the same observations and the same values"""

        return self.to_pandas().equals(other.to_pandas())
