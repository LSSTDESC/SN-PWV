"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). Here we demonstrate running a pipeline
synchronously.

.. doctest:: python

   >>> from snat_sim.pipeline import FittingPipeline

   >>> pipeline = FittingPipeline(
   >>>     cadence='alt_sched',
   >>>     sim_model=SNModel('salt2'),
   >>>     fit_model=SNModel('salt2'),
   >>>     vparams=['x0', 'x1', 'c'],
   >>>     out_path='./demo_out_path.csv',
   >>>     pool_size=6
   >>> )

Module Docs
-----------
"""

from __future__ import annotations

import warnings
from copy import copy
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import *
from typing import List, Dict, Iterable

import sncosmo
from astropy.table import Table
from egon.connectors import Output, Input
from egon.nodes import Source, Node, Target
from egon.pipeline import Pipeline

from . import plasticc, reference_stars, constants as const
from .models import SNModel, PWVModel, ObservedCadence

warnings.simplefilter('ignore', category=DeprecationWarning)


@dataclass
class PipelineResult:
    """Class representation of data products produced by the ``FittingPipeline``"""

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
            A list of strings and floats
        """

        out_list = self.to_list(fit_params, sim_params)
        return ','.join(map(str, out_list)) + '\n'

    def to_list(self, fit_params, sim_params) -> List[str, float]:
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


class LoadPlasticcSims(Source):
    """Pipeline node for loading PLaSTICC data from disk

    Connectors:
        lc_output: The loaded PLaSTICC light-curves as ``astropy.Table`` objects
    """

    def __init__(self, cadence: str, model: int = 11, iter_lim: int = float('inf'), num_processes: int = 1) -> None:
        """Source node for loading PLaSTICC light-curves from disk

        Args:
            cadence: Cadence to use when simulating light-curves
            model: The PLaSTICC supernova model to load simulation for (Default is model 11 - Normal SNe)
            iter_lim: Exit after loading the given number of light-curves
            num_processes: Number of processes to allocate to the node
        """

        self.cadence = cadence
        self.model = model
        self.iter_lim = iter_lim

        # Node connectors
        self.lc_output = Output()
        super().__init__(num_processes)

    def action(self) -> None:
        """Load PLaSTICC light-curves from disk"""

        light_curve_iter = plasticc.iter_lc_for_cadence_model(self.cadence, model=self.model)
        for i, light_curve in enumerate(light_curve_iter):
            if i >= self.iter_lim:
                break

            self.lc_output.put(light_curve)


class SimulateLightCurves(Node):
    """Pipeline node for simulating light-curves based on PLaSTICC cadences

    Connectors:
        plasticc_data_input: PLaSTICC light-curves as ``astropy.Table`` objects
        simulation_output: Simulated light-curves as  ``astropy.Table`` objects
    """

    def __init__(
            self,
            sn_model: SNModel,
            ref_stars: Collection[str] = None,
            pwv_model: PWVModel = None,
            num_processes: int = 1,
            abs_mb: float = const.betoule_abs_mb,
            cosmo=const.betoule_cosmo
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when simulating light-curves
            ref_stars: List of reference star types to calibrate simulated supernova with
            pwv_model: Model for the PWV concentration the reference stars are observed at
            num_processes: Number of processes to allocate to the node
            abs_mb: The absolute B-band magnitude of the simulated SNe
            cosmo: Cosmology to assume in the simulation
        """

        self.sim_model = sn_model
        self.ref_stars = ref_stars
        self.pwv_model = pwv_model
        self.abs_mb = abs_mb
        self.cosmo = cosmo

        # Node connectors
        self.plasticc_data_input = Input()
        self.simulation_output = Output()
        self.failure_result_output = Output()
        super().__init__(num_processes)

    def duplicate_plasticc_lc(self, plasticc_lc: Table, zp: float = 30) -> Table:
        """Duplicate a plastic light-curve using the simulation model

        Args:
            plasticc_lc: The light-curve to duplicate
            zp: Zero-point of the duplicated light-curve
        """

        # Get simulation parameters and observational cadence
        params, plasticc_cadence = ObservedCadence.from_plasticc(plasticc_lc, zp=zp)

        # Set model parameters and scale the source brightness to the desired intrinsic brightness
        model_for_sim = copy(self.sim_model)
        model_for_sim.update({p: v for p, v in params.items() if p in model_for_sim.param_names})
        model_for_sim.set_source_peakabsmag(self.abs_mb, 'standard::b', 'AB', cosmo=self.cosmo)

        # Simulate the light-curve. Make sure to include model parameters as meta data
        duplicated = model_for_sim.simulate_lc(plasticc_cadence)
        duplicated.meta = params
        duplicated.meta['x0'] = model_for_sim['x0']

        # Rescale the light-curve using the reference star catalog if provided
        if self.ref_stars is not None:
            pwv_los = self.pwv_model.pwv_los(
                duplicated['time'],
                ra=plasticc_lc.meta['RA'],
                dec=plasticc_lc.meta['DECL'],
                time_format='mjd')

            duplicated = reference_stars.divide_ref_from_lc(duplicated, pwv_los, self.ref_stars)

        return duplicated

    def action(self) -> None:
        """Simulate light-curves with atmospheric effects"""

        for light_curve in self.plasticc_data_input.iter_get():
            try:
                duplicated_lc = self.duplicate_plasticc_lc(light_curve, zp=30)

            except Exception as e:
                params, _ = ObservedCadence.from_plasticc(light_curve)  # Format params to match the simulation model
                result = PipelineResult(light_curve.meta['SNID'], sim_params=params, message=str(e))
                self.failure_result_output.put(result)

            else:
                self.simulation_output.put(duplicated_lc)


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

        return sncosmo.fit_lc(
            light_curve, self.sn_model, self.vparams, bounds=self.bounds,
            guess_t0=False, guess_amplitude=False, guess_z=False, warn=False)

    def action(self) -> None:
        """Fit light-curves"""

        for light_curve in self.light_curves_input.iter_get():
            # Use the true light-curve parameters as the initial guess
            self.sn_model.update({k: v for k, v in light_curve.meta.items() if k in self.sn_model.param_names})

            try:
                fitted_result, fitted_model = self.fit_lc(light_curve)

            except Exception as excep:
                self.fit_results_output.put(
                    PipelineResult(snid=light_curve.meta['SNID'], sim_params=light_curve.meta,
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
                        mb=fitted_model.mB(),
                        abs_mag=fitted_model.MB(),
                        message=f'{self.__class__.__name__}: {fitted_result.message}'
                    )
                )


class FitResultsToDisk(Target):
    """Pipeline node for writing fit results to disk

    Connectors:
        fit_results_input: List of values to write as single line in CSV format
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

        self.out_path.parent.mkdir(exist_ok=True, parents=True)

        column_names = PipelineResult.column_names(self.sim_model.param_names, self.fit_model.param_names)
        with self.out_path.open('w') as outfile:
            outfile.write(','.join(column_names))
            outfile.write('\n')

    def action(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open('a') as outfile:
            for result in self.fit_results_input.iter_get():
                outfile.write(result.to_csv(self.sim_model.param_names, self.fit_model.param_names))


class FittingPipeline(Pipeline):
    """Pipeline of parallel processes for simulating and fitting light-curves"""

    def __init__(
            self,
            cadence: str,
            sim_model: SNModel,
            fit_model: SNModel,
            vparams: List[str],
            out_path: Union[str, Path],
            fitting_pool: int = 1,
            simulation_pool: int = 1,
            bounds: Dict[str, Tuple[Number, Number]] = None,
            max_queue: int = 100,
            iter_lim: int = float('inf'),
            ref_stars: Collection[str] = None,
            pwv_model: PWVModel = None
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            out_path: Path to write results to (.csv extension is enforced)
            fitting_pool: Number of child processes allocated to simulating light-curves
            simulation_pool: Number of child processes allocated to fitting light-curves
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            ref_stars: List of reference star types to calibrate simulated supernova with
            pwv_model: Model for the PWV concentration the reference stars are observed at
        """

        if ref_stars and (pwv_model is None):
            raise ValueError('Cannot perform reference star subtraction without ``pwv_model`` argument')

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcSims(cadence=cadence, iter_lim=iter_lim)
        self.simulate_light_curves = SimulateLightCurves(
            sn_model=sim_model,
            ref_stars=ref_stars,
            pwv_model=pwv_model,
            num_processes=simulation_pool)

        self.fit_light_curves = FitLightCurves(
            sn_model=fit_model, vparams=vparams, bounds=bounds, num_processes=fitting_pool)

        self.write_to_disk = FitResultsToDisk(sim_model, fit_model, out_path)

        # Connect pipeline nodes together
        self.load_plastic.lc_output.connect(self.simulate_light_curves.plasticc_data_input)
        self.simulate_light_curves.simulation_output.connect(self.fit_light_curves.light_curves_input)
        self.simulate_light_curves.failure_result_output.connect(self.write_to_disk.fit_results_input)
        self.fit_light_curves.fit_results_output.connect(self.write_to_disk.fit_results_input)

        if max_queue:  # Limit the number of light-curves fed into the pipeline
            self.simulate_light_curves.plasticc_data_input.maxsize = max_queue

        super(FittingPipeline, self).__init__()
