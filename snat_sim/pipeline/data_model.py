from copy import copy
from typing import List

import numpy as np
import sncosmo

from snat_sim.models import SNModel
from .. import constants as const


class DataModel:
    """Enforces the data model of pipeline output files"""

    def __init__(self, sim_model: SNModel, fit_model: SNModel) -> None:
        """Formats data from a given supernova model into a coherent tabular format

        Args:
            fit_model (Model): Supernova model to reflect in the outputted data model
        """

        self.sim_model = copy(sim_model)
        self.fit_model = copy(fit_model)

    def build_table_entry(
            self, meta: dict, fitted_model: SNModel, result: sncosmo.utils.Result) -> List:
        """Combine light-curve fit results into single row matching the output table file format

        Args:
            meta: Meta data for the simulated light-curve
            fitted_model: Supernova model fitted to the data
            result: sncosmo fit Result object

        Returns:
            A list of strings and floats
        """

        out_list = [meta['SNID']]
        out_list.extend(meta.get(param, -99) for param in self.sim_model.param_names)
        out_list.extend(result.parameters)
        out_list.extend(result.errors.values())
        out_list.append(result.chisq)
        out_list.append(result.ndof)
        out_list.append(fitted_model.bandmag('bessellb', 'ab', time=fitted_model['t0']))
        out_list.append(fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo))
        out_list.append(result.message)
        return out_list

    def build_masked_entry(self, meta: dict, exception: Exception) -> List:
        """Create a masked table entry for a failed light-curve fit

        Args:
            meta: Meta data for the simulated light-curve
            exception: Exception raised by the failed fit

        Returns:
            A list of strings and floats with masked values set as -99
        """

        out_list = [meta['SNID']]
        out_list.extend(meta.get(param, -99) for param in self.sim_model.param_names)

        remaining_columns = len(self.column_names) - len(out_list)
        out_list.extend(np.full(remaining_columns - 1, -99))
        out_list.append(str(exception))
        return out_list

    @property
    def column_names(self) -> List[str]:
        """Return a list of column names for the data model

        Returns:
            List of column names as strings
        """

        col_names = ['SNID']
        col_names.extend('sim_' + param for param in self.sim_model.param_names)
        col_names.extend(self.fit_model.param_names)
        col_names.extend(param + '_err' for param in self.fit_model.param_names)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')
        return col_names
