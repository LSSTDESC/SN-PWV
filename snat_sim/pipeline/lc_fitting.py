from __future__ import annotations

import warnings
from copy import copy
from pathlib import Path
from typing import *
from typing import Dict, List

import sncosmo
from astropy.table import Table
from egon.connectors import Input, Output
from egon.nodes import Node, Target

from .data_model import PipelineResult
from .. import constants as const
from ..models import SNModel

warnings.simplefilter('ignore', category=DeprecationWarning)


class FitLightCurves(Node):
    """Pipeline node for fitting simulated light-curves

    Connectors:
        light_curves_input: Light-curves to fit
        fit_results_output: Fit results as a list
    """

    def __init__(
            self,
            sn_model: SNModel,
            vparams: List[str],
            bounds: Dict = None,
            num_processes: int = 1
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            num_processes: Number of processes to allocate to the node
        """

        self.sn_model = sn_model
        self.vparams = vparams
        self.bounds = bounds

        # Node Connectors
        self.light_curves_input = Input()
        self.fit_results_output = Output()
        super(FitLightCurves, self).__init__(num_processes)

    def fit_lc(self, light_curve: Table):
        """Fit the given light-curve

        Args:
            A light-curve in ``sncosmo`` format

        Returns:
            - The optimization result represented as a ``Result`` object
            - A copy of ``self.model`` with parameter values set to optimize the chi-square
        """

        # Use the true light-curve parameters as the initial guess
        model = copy(self.sn_model)
        model.update({k: v for k, v in light_curve.meta.items() if k in self.sn_model.param_names})

        return sncosmo.fit_lc(
            light_curve, model, self.vparams, bounds=self.bounds,
            guess_t0=False, guess_amplitude=False, guess_z=False, warn=False)

    def action(self) -> None:
        """Fit light-curves"""

        for light_curve in self.light_curves_input.iter_get():
            try:
                fitted_result, fitted_model = self.fit_lc(light_curve)

            except Exception as excep:
                self.fit_results_output.put(
                    PipelineResult(
                        snid=light_curve.meta['SNID'], sim_params=light_curve.meta,
                        message=f'{self.__class__.__name__}: {excep}')
                )

            else:
                self.fit_results_output.put(
                    PipelineResult(
                        snid=light_curve.meta['SNID'],
                        sim_params=light_curve.meta,
                        fit_params=dict(zip(fitted_result.param_names, fitted_result.parameters)),
                        fit_err=fitted_result.errors,
                        chisq=fitted_result.chisq,
                        ndof=fitted_result.ndof,
                        mb=fitted_model.source.bandmag('bessellb', 'ab', phase=0),
                        abs_mag=fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo),
                        message=f'{self.__class__.__name__}: {fitted_result.message}'
                    )
                )


class FitResultsToDisk(Target):
    """Pipeline node for writing fit results to disk

    Connectors:
        fit_results_input: ``PipelineResult`` objects to write as individual lines in CSV format
    """

    def __init__(
            self, sim_model: SNModel, fit_model: SNModel, out_path: Union[str, Path], num_processes: int = 1
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            out_path: Path to write results to (.csv extension is enforced)
        """

        self.sim_model = sim_model
        self.fit_model = fit_model
        self.out_path = Path(out_path)

        # Node connectors
        self.fit_results_input = Input()
        super(FitResultsToDisk, self).__init__(num_processes)

    def setup(self) -> None:
        """Ensure the parent directory of the destination file exists"""

        self.out_path.parent.mkdir(exist_ok=True, parents=False)

        column_names = PipelineResult.column_names(self.sim_model.param_names, self.fit_model.param_names)
        with self.out_path.open('w') as outfile:
            outfile.write(','.join(column_names))
            outfile.write('\n')

    def action(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open('a') as outfile:
            for result in self.fit_results_input.iter_get():
                result.message = result.message.replace('\n', ' ').replace(',', '')
                outfile.write(result.to_csv(self.sim_model.param_names, self.fit_model.param_names))
