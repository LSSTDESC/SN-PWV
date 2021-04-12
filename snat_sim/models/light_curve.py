"""Abstract representations of astronomical time-series data

The ``LightCurve`` class represents astronomical light-curves and
provides an easy interface for casting the data into other object
types commonly used in third party astronomical and data-analysis software.

.. doctest:: python

   >>> from snat_sim.models import LightCurve

   >>> light_curve = LightCurve(
   ... time=[55070.000000, 55072.051282, 55074.102564, 55076.153846],
   ... band=['sdssg', 'sdssr', 'sdssi', 'sdssz'],
   ... flux=[0.363512, -0.200801,  0.307494,  1.087761],
   ... fluxerr=[0.672844, 0.672844, 0.672844, 0.672844],
   ... zp=[25.0, 25.0, 25.0, 25.0],
   ... zpsys=['ab', 'ab', 'ab', 'ab'])

"""

from typing import *

import numpy as np
import pandas as pd
from astropy.table import Table

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
            zpsys: Collection[str]
    ) -> None:
        """An astronomical light-curve

        Args:
            time: Time values for each observation in a numerical format (e.g. JD or MJD)
            band: The band each observation was performed in
            flux: The flux of each observation
            fluxerr: The error in each flux value
            zp: The zero-point of each observation
            zpsys: The zero-point system of each observation
        """

        self.time = time
        self.band = band
        self.flux = flux
        self.fluxerr = fluxerr
        self.zp = zp
        self.zpsys = zpsys

        if np.diff([len(attr) for attr in (time, band, flux, fluxerr, zp, zpsys)]).any():
            raise ValueError('Argument lengths are not all the same.')

    def to_dict(self):
        """Return the light_curve data as a dictionary of it's attributes

        Returns:
            A dictionary with the light-curve data
        """

        return {
            'time': self.time,
            'band': self.band,
            'flux': self.flux,
            'zp': self.zp,
            'zpsys': self.zpsys
        }

    def to_sncosmo(self) -> Table:
        """Return the light-curve data as a Table formatted for use with sncosmo

        Returns:
            A table formatted for use with sncosmo
        """

        return Table(self.to_dict())

    def to_pandas(self) -> pd.DataFrame:
        """Return the light-curve data as a pandas DataFrame

        Returns:
            A ``pandas.DataFrame`` instance with the light-curve data
        """

        return pd.DataFrame(self.to_dict())
