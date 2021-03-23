"""Defines a standardized data model for communication between pipeline nodes.

Usage Example
-------------

Data models are defined as Python data classes. All fields in the data model
are onptional except for the supernova identifier (``snid``). Other fields
include the supernova model parameters used in a light-curve simulation / fit
and the chi-squared, degrees of freedom, and B-band magnitudes returned by
the fitted model.

.. doctest::

   >>> from snat_sim.pipeline.data_model import PipelinePacket
   >>> data_obj = PipelinePacket(
   ... snid='1234567',
   ... sim_params={'x0': 1, 'x1': .1, 'c': .5},
   ... fit_params={'x0': .9, 'x1': .12, 'c': .51},
   ... fit_err={'x0': .1, 'x1': .01, 'c': .05},
   ... chisq=12,
   ... ndof=11,
   ... mb=22.5,
   ... abs_mag=-19.1,
   ... message='The fit exited successfully'
   ... )

Data products can be converted into familiar data structures using instance
the methods demonstrated below (see the full class documentation for a
complete list of available methods). Missing numerical data is masked using
the value ``-99.99``.

.. doctest::

   >>> # Pick which simulated and fitted parameters to include in the output
   >>> include_sim_params = ['x0', 'x1']
   >>> include_fit_params = ['x0', 'c']
   >>>
   >>> data_list = data_obj.to_list(include_sim_params, include_fit_params)
   >>> data_str = data_obj.to_csv(include_sim_params, include_fit_params)

Module Docs
-----------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .. import constants as const
from .. import types
from ..models import ObservedCadence, SNFitResult, SNModel


@dataclass
class PipelinePacket:
    """Class representation of internal pipeline data products"""

    snid: str
    sim_params: Optional[types.NumericalParams] = None
    cadence: Optional[ObservedCadence] = None
    light_curve: Optional[pd.DataFrame] = None
    sim_data: Optional[pd.DataFrame] = None
    fit_result: Optional[SNFitResult] = None
    fitted_model: Optional[SNModel] = None
    message: Optional[str] = None

    def fit_result_to_pandas(self) -> pd.DataFrame:
        col_names = ['snid']
        col_names.extend('fit_' + param for param in self.sim_params)
        col_names.extend('err_' + param for param in self.sim_params)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')

        data_list = [self.snid]
        if None not in (self.sim_params, self.fit_result, self.fitted_model):
            data_list.extend(self.fit_result.parameters)
            data_list.extend(self.fit_result.errors.get(p, -99.99) for p in self.sim_params)
            data_list.append(self.fit_result.chisq)
            data_list.append(self.fit_result.ndof)
            data_list.append(self.fitted_model.source.bandmag('bessellb', 'ab', phase=0))
            data_list.append(self.fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo))
            data_list.append(self.message)

        else:
            non_snid_columns = 2 * len(self.sim_params) + 4
            data_list.extend(-99.99 for _ in range(non_snid_columns))
            data_list.append("")
        print('---------\n\n\n', pd.DataFrame(pd.Series(data_list, index=col_names)).T, '\n\n\n---------')
        return pd.DataFrame(pd.Series(data_list, index=col_names)).T

    def sim_params_to_pandas(self)-> pd.DataFrame:
        return pd.DataFrame(pd.Series(self.sim_params)).T
