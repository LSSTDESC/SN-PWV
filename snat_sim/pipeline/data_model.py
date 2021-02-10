"""Defines a standardized data model for communication between pipeline nodes

Usage Example
-------------

Data models are defined as Python data classes. All fields in the data model
are onptional except for the supernova identifier (``snid``). Other fields
include the supernova model parameters used in a light-curve simulation / fit
and the chi-squared, degrees of freedom, and B-band magnitudes returned by
the fitted model.

.. doctest::

   >>> from snat_sim.pipeline.data_model import PipelineResult
   >>> data_obj = PipelineResult(
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

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class PipelineResult:
    """Class representation of internal pipeline data products"""

    snid: str
    sim_params: Dict[str, float] = field(default_factory=dict)
    fit_params: Dict[str, float] = field(default_factory=dict)
    fit_err: Dict[str, float] = field(default_factory=dict)
    chisq: float = -99.99
    ndof: int = -99.99
    mb: float = -99.99
    abs_mag: float = -99.99
    message: str = ''

    def to_csv(self, sim_params: Iterable[str], fit_params: Iterable[str]) -> str:
        """Combine light-curve fit results into single row matching the output table file format

        Args:
            sim_params: The simulated parameter values to include int the output
            fit_params: The fitted parameter values to include in the output

        Returns:
            A string with data in CSV format
        """

        out_list = self.to_list(sim_params, fit_params)
        return ','.join(map(str, out_list)) + '\n'

    def to_list(self, sim_params, fit_params) -> List[str, float]:
        """Return class data as a list with missing values masked as -99.99

        Args:
            sim_params: The order of the simulated parameter values in the return
            fit_params: The order of the fitted parameter values in the return

        Returns:
            A list of strings and floats
        """

        out_list = [self.snid]
        out_list.extend(self.sim_params.get(param, -99.99) for param in sim_params)
        out_list.extend(self.fit_params.get(param, -99.99) for param in fit_params)
        out_list.extend(self.fit_err.get(param, -99.99) for param in fit_params)
        out_list.append(self.chisq)
        out_list.append(self.ndof)
        out_list.append(self.mb)
        out_list.append(self.abs_mag)
        out_list.append(self.message)
        return out_list

    @staticmethod
    def column_names(sim_params: Iterable[str], fit_params: Iterable[str]) -> List[str]:
        """Return a list of column names matching the data model used by ``PipelineResult.to_csv``

        Args:
            sim_params: The simulated parameter values to include int the output
            fit_params: The fitted parameter values to include in the output

        Returns:
            List of column names as strings
        """

        col_names = ['snid']
        col_names.extend('sim_' + param for param in sim_params)
        col_names.extend('fit_' + param for param in fit_params)
        col_names.extend('err_' + param for param in fit_params)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')
        return col_names
