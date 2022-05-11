"""Defines a standardized data model for communication between pipeline nodes.

Usage Example
-------------

Data models are defined as Python data classes. All fields in the data model
are onptional except for the supernova identifier (``snid``). Other fields
include the supernova model parameters used in a light-curve simulation / fit
and the cadence used in the simulation.

.. doctest::

   >>> import sncosmo
   >>> from snat_sim.models import SNModel, LightCurve
   >>> from snat_sim.pipeline.data_model import PipelinePacket

   >>> # Load example data and an example supernova model
   >>> example_data = sncosmo.load_example_data()
   >>> light_curve = LightCurve.from_sncosmo(example_data)
   >>> model_parameters = example_data.meta
   >>> sn_model = SNModel('Salt2')

   >>> # Set an initial guess for fitting the model parameters
   >>> sn_model.update(model_parameters)

   >>> fit_result = sn_model.fit_lc(light_curve, vparam_names=['x0', 'x1', 'c'])
   >>> packet = PipelinePacket(
   ...     snid=1234,                    # Unique SN identifier
   ...     sim_params=model_parameters,  # Parameters used to simulate the light-curve
   ...     light_curve=light_curve,      # The simulated light-curve
   ...     fit_result=fit_result,        # ``SNFitResult`` object
   ...     message='This fit was a success!'
   ... )

Module Docs
-----------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .. import types
from ..models import LightCurve, ObservedCadence, SNFitResult, SNModel

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
    light_curve: Optional[LightCurve] = None
    fit_result: Optional[SNFitResult] = None
    covariance: Optional[pd.DataFrame] = None
    message: Optional[str] = ""

    def sim_params_to_pandas(self) -> pd.DataFrame:
        """Return simulated parameters as a pandas Dataframe

        Return:
            Parameters used in the simulation of light-curves
        """

        out_data = pd.Series(self.sim_params)
        out_data['snid'] = self.snid
        return pd.DataFrame(out_data).T

    def fitted_params_to_pandas(self) -> pd.DataFrame:
        """Return fitted parameters as a pandas Dataframe

        Return:
            Parameters recovered from fitting a light-curve
        """

        if self.fit_result is None:
            raise ValueError('Fit results are not stored in the data packet.')

        col_names = ['snid']
        col_names.extend('fit_' + param for param in self.fit_result.param_names)
        col_names.extend('err_' + param for param in self.fit_result.param_names)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('apparent_bessellb')
        col_names.append('absolute_bessellb')
        # col_names.append('message')

        data_list = [self.snid]
        data_list.extend(self.fit_result.parameters)
        data_list.extend(self.fit_result.errors.get(p, MASK_VALUE) for p in self.fit_result.param_names)
        data_list.append(self.fit_result.chisq)
        data_list.append(self.fit_result.ndof)
        data_list.append(self.fit_result.apparent_bessellb)
        data_list.append(self.fit_result.absolute_bessellb)
        return pd.DataFrame(pd.Series(data_list, index=col_names)).T

    def packet_status_to_pandas(self) -> pd.DataFrame:
        """Return the packet status message as a pandas ``DataFrame``

        Return:
            Dataframe with the snid, fit success status, and result message
        """

        success = self.fit_result.success if self.fit_result else False
        return pd.DataFrame({'snid': [self.snid], 'success': [success], 'message': [self.message[:150]]})
