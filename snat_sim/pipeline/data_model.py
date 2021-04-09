"""Defines a standardized data model for communication between pipeline nodes.

Usage Example
-------------

Data models are defined as Python data classes. All fields in the data model
are onptional except for the supernova identifier (``snid``). Other fields
include the supernova model parameters used in a light-curve simulation / fit
and the cadence used in the simulation.

.. doctest::

   >>> import sncosmo
   >>> from snat_sim.models import SNModel
   >>> from snat_sim.pipeline.data_model import PipelinePacket
   
   >>> # Load example data and an example supernova model
   >>> data = sncosmo.load_example_data()
   >>> sn_model = SNModel('Salt2')

   >>> # Set an initial guess for fitting the model parameters
   >>> sn_model.update(data.meta)

   >>> fit_result, fitted_model = sn_model.fit_lc(data, vparam_names=['x1', 'c'])
   >>> packet = PipelinePacket(
   ...     snid=1234,                  # Unique SN identifier
   ...     sim_params=data.meta,       # Parameters used to simulate the light-curve
   ...     light_curve=data,           # The simulated light-curve
   ...     fit_result=fit_result,      # ``SNFitResult`` object
   ...     fitted_model=fitted_model,  # The fitted model (set to best fit parameter values)
   ...     message='This fit was a success!'
   ... )

The simulation and fittd parameters can be converted into pandas Dataframes.
Missing numerical data is masked using the value ``-99.99``.

.. doctest::

   >>> # Pick which simulated and fitted parameters to include in the output
   >>> include_sim_params = ['x0', 'x1']
   >>> include_fit_params = ['x0', 'c']
   >>>
   >>> packet.sim_params_to_pandas()
       x1    c    z        x0       t0    SNID
   0  0.5  0.2  0.5  0.000012  55100.0  1234.0

   >>> packet.fitted_params_to_pandas()
      snid fit_z   fit_t0    fit_x0    fit_x1     fit_c  err_z err_t0 err_x0   err_x1     err_c      chisq ndof        mb    abs_mag                  message
   0  1234   0.5  55100.0  0.000012  0.359913  0.209899 -99.99 -99.99 -99.99  0.26414  0.020136  36.071269   38  22.80058 -19.467872  This fit was a success!

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

MASK_VALUE = -99.99


@dataclass
class PipelinePacket:
    """Class representation of internal pipeline data products

    Fields:
        snid: Unique identifier for the data packet
        sim_params: Parameters used to simulate the light-curve
        cadence: The observational cadence used to simulate the light-curve
        light_curve: The simulated light-curve
        fit_result: Fit result from fitting the light-curve
        fitted_model: Model used to fit the light-curve
        message: Status message
    """

    snid: int
    sim_params: Optional[types.NumericalParams] = None
    cadence: Optional[ObservedCadence] = None
    light_curve: Optional[pd.DataFrame] = None
    fit_result: Optional[SNFitResult] = None
    fitted_model: Optional[SNModel] = None
    message: Optional[str] = ""

    def sim_params_to_pandas(self) -> pd.DataFrame:
        """Return simulated parameters as a pandas Dataframe

        Return:
            Parameters used in the simulation of light-curves
        """

        out_data = pd.Series(self.sim_params)
        out_data['SNID'] = self.snid
        out_data['message'] = self.message
        return pd.DataFrame(out_data).T

    def fitted_params_to_pandas(self) -> pd.DataFrame:
        """Return fitted parameters as a pandas Dataframe

        Return:
            Parameters recovered from fitting a light-curve
        """

        if None in (self.fit_result, self.fitted_model):
            raise ValueError('Fit results are not stored in the data packet.')

        col_names = ['snid']
        col_names.extend('fit_' + param for param in self.fit_result.param_names)
        col_names.extend('err_' + param for param in self.fit_result.param_names)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')

        data_list = [self.snid]
        data_list.extend(self.fit_result.parameters)
        data_list.extend(self.fit_result.errors.get(p, MASK_VALUE) for p in self.fit_result.param_names)
        data_list.append(self.fit_result.chisq)
        data_list.append(self.fit_result.ndof)
        data_list.append(self.fitted_model.source.bandmag('bessellb', 'ab', phase=0))
        data_list.append(self.fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo))
        data_list.append(self.message)
        return pd.DataFrame(pd.Series(data_list, index=col_names)).T
