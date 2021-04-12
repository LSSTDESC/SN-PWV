"""Abstract representations of astronomical time-series data."""

from __future__ import annotations

from copy import copy
from typing import *
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.table import Table

if TYPE_CHECKING:
    from ..types import Numeric


class LightCurve:
    """Abstract representation of an astronomical light-curve."""

    def __init__(
            self,
            time: Collection[Numeric],
            band: Collection[str],
            flux: Collection[Numeric],
            fluxerr: Collection[Numeric],
            zp: Collection[Numeric],
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

        try:
            phot_flag = data['phot_flag']

        except KeyError:
            phot_flag = None

        return LightCurve(
            time=data['time'],
            band=data['band'],
            flux=data['flux'],
            fluxerr=data['fluxerr'],
            zp=data['zp'],
            zpsys=data['zpsys'],
            phot_flag=phot_flag
        )

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
        ))

    def __len__(self) -> int:
        """The number of observations in the light-curve"""

        return len(self.band)

    def copy(self) -> LightCurve:
        """Return a copy of the instance"""

        return copy(self)

    def __eq__(self, other: LightCurve):
        """Check whether both objects have the same observations and the same values"""

        return self.to_pandas().equals(other.to_pandas())
