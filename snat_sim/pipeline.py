"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). A pipeline instance can be created as
follows:

.. doctest:: python

   >>> from snat_sim.models import SNModel
   >>> from snat_sim.pipeline import FittingPipeline

   >>> pipeline = FittingPipeline(
   ...     cadence='alt_sched',
   ...     sim_model=SNModel('salt2'),
   ...     fit_model=SNModel('salt2'),
   ...     vparams=['x0', 'x1', 'c'],
   ...     out_path='./demo_out_path.csv',
   ...     fitting_pool=6,
   ...     simulation_pool=3
   ... )

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
from typing import Dict, Iterable, List

import sncosmo
from astropy.table import Table
from egon.connectors import Input, Output
from egon.nodes import Node, Source, Target
from egon.pipeline import Pipeline

from . import constants as const
from .models import AbstractVariablePWVEffect, ObservedCadence, PWVModel, SNModel, StaticPWVTrans
from .plasticc import PLaSTICC
from .reference_stars import VariableCatalog

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
        output: Tuple with the simulation params (``dict``) and cadence (``ObservedCadence``)
    """

    def __init__(self, cadence: str, model: int = 11, iter_lim: int = float('inf'), num_processes: int = 1) -> None:
        """Source node for loading PLaSTICC light-curves from disk

        Args:
            cadence: Cadence to use when simulating light-curves
            model: The PLaSTICC supernova model to load simulation for (Default is model 11 - Normal SNe)
            iter_lim: Exit after loading the given number of light-curves
            num_processes: Number of processes to allocate to the node
        """

        self.cadence = PLaSTICC(cadence, model)
        self.iter_lim = iter_lim

        # Node connectors
        self.output = Output()
        super().__init__(num_processes)

    def action(self) -> None:
        """Load PLaSTICC light-curves from disk"""

        for light_curve in self.cadence.iter_lc(iter_lim=self.iter_lim):
            self.output.put(ObservedCadence.from_plasticc(light_curve))


class SimulateLightCurves(Node):
    """Pipeline node for simulating light-curves based on PLaSTICC cadences

    Connectors:
        plasticc_data_input: PLaSTICC light-curves as ``astropy.Table`` objects
        simulation_output: Simulated light-curves as  ``astropy.Table`` objects
    """

    def __init__(
            self,
            sn_model: SNModel,
            catalog: VariableCatalog = None,
            num_processes: int = 1,
            abs_mb: float = const.betoule_abs_mb,
            cosmo=const.betoule_cosmo,
            include_pwv: bool = False
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when simulating light-curves
            catalog: Optional reference start catalog to calibrate simulated flux values to
            num_processes: Number of processes to allocate to the node
            abs_mb: The absolute B-band magnitude of the simulated SNe
            cosmo: Cosmology to assume in the simulation
        """

        self.sim_model = sn_model
        self.catalog = catalog
        self.abs_mb = abs_mb
        self.cosmo = cosmo
        self.include_pwv_col = include_pwv

        # Node connectors
        self.plasticc_data_input = Input()
        self.simulation_output = Output()
        self.failure_result_output = Output()
        super().__init__(num_processes)

    def duplicate_plasticc_lc(self, params: Dict[str, float], cadence: ObservedCadence) -> Table:
        """Duplicate a plastic light-curve using the simulation model

        Args:
            params: The simulation parameters to use with ``self.model``
            cadence: The observed cadence of the returned light-curve
        """

        # Set model parameters and scale the source brightness to the desired intrinsic brightness
        model_for_sim = copy(self.sim_model)
        model_for_sim.update({p: v for p, v in params.items() if p in model_for_sim.param_names})
        model_for_sim.set_source_peakabsmag(self.abs_mb, 'standard::b', 'AB', cosmo=self.cosmo)

        # Simulate the light-curve. Make sure to include model parameters as meta data
        duplicated = model_for_sim.simulate_lc(cadence)
        duplicated.meta = params
        duplicated.meta.update(dict(zip(model_for_sim.param_names, model_for_sim.parameters)))

        # Rescale the light-curve using the reference star catalog if provided
        if self.catalog is not None:
            duplicated = self.catalog.calibrate_lc(
                duplicated, duplicated['time'], ra=params['ra'], dec=params['dec'])

        # Add the simulated PWV concentration if there is a variable PWV transmission effect.
        if self.include_pwv_col:
            self.add_pwv_columns_to_table(duplicated, model_for_sim, ra=params['ra'], dec=params['dec'])

        return duplicated

    @staticmethod
    def add_pwv_columns_to_table(light_curve: Table, model_for_sim: SNModel, ra: float, dec: float) -> None:
        """Add columns for PWV and Airmass to a light-curve table

        Args:
            light_curve: The simulated light-curve table
            model_for_sim: The model used to simulate the light-curve
            ra: The Right Ascension of the supernova
            dec: The declination  of the supernova
        """

        for effect in model_for_sim.effects:
            if isinstance(effect, AbstractVariablePWVEffect):
                light_curve['pwv'] = effect.assumed_pwv(light_curve['time'])
                light_curve['airmass'] = PWVModel.calc_airmass(light_curve['time'], ra=ra, dec=dec)
                break

            if isinstance(effect, StaticPWVTrans):
                light_curve['pwv'] = effect['pwv']
                light_curve['airmass'] = 1
                break

    def action(self) -> None:
        """Simulate light-curves with atmospheric effects"""

        for params, cadence in self.plasticc_data_input.iter_get():
            try:
                duplicated_lc = self.duplicate_plasticc_lc(params, cadence)

            except Exception as e:
                result = PipelineResult(
                    params['SNID'], sim_params=params,
                    message=f'{self.__class__.__name__}: {e}')
                self.failure_result_output.put(result)

            else:
                self.simulation_output.put(duplicated_lc)


class SimulationToDisk(Target):
    """Pipeline node for writing simulated light-curves to disk

    Connectors:
        simulation_input: Simulated light-curves
    """

    def __init__(
            self, out_dir: Union[str, Path], num_processes: int = 1
    ) -> None:
        """Write simulated light-curves to disk

        Args:
            out_dir: Path to write results to (.csv extension is enforced)
        """

        self.out_dir = Path(out_dir)
        self.simulation_input = Input()
        super(SimulationToDisk, self).__init__(num_processes)

    def setup(self) -> None:
        """Ensure the output directory exists"""

        self.out_dir.mkdir(exist_ok=True, parents=False)

    def action(self) -> None:
        """Write simulated light-curves to disk"""

        for lc in self.simulation_input.iter_get():
            path = (self.out_dir / lc.meta['SNID']).with_suffix('.ecsv')
            lc.write(path, overwrite=True)


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


class FittingPipeline(Pipeline):
    """Pipeline of parallel processes for simulating and fitting light-curves"""

    def __init__(
            self,
            cadence: str,
            sim_model: SNModel,
            fit_model: SNModel,
            vparams: List[str],
            out_path: Union[str, Path],
            sim_dir: Union[str, Path] = None,
            fitting_pool: int = 1,
            simulation_pool: int = 1,
            bounds: Dict[str, Tuple[Number, Number]] = None,
            max_queue: int = 200,
            iter_lim: int = float('inf'),
            catalog: VariableCatalog = None
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            out_path: Path to write results to (.csv extension is enforced)
            sim_dir: Optionally write simulated light-curves to disk in the given directory as individual ecsv files
            fitting_pool: Number of child processes allocated to simulating light-curves
            simulation_pool: Number of child processes allocated to fitting light-curves
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            catalog: Reference star catalog to calibrate simulated supernova with
        """

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcSims(cadence, model=11, iter_lim=iter_lim)

        self.simulate_light_curves = SimulateLightCurves(
            sn_model=sim_model,
            catalog=catalog,
            num_processes=simulation_pool,
            include_pwv=sim_dir is not None
        )

        self.fit_light_curves = FitLightCurves(
            sn_model=fit_model, vparams=vparams, bounds=bounds, num_processes=fitting_pool)

        self.fits_to_disk = FitResultsToDisk(sim_model, fit_model, out_path)

        # Connect pipeline nodes together
        self.load_plastic.output.connect(self.simulate_light_curves.plasticc_data_input)
        self.simulate_light_curves.simulation_output.connect(self.fit_light_curves.light_curves_input)
        self.simulate_light_curves.failure_result_output.connect(self.fits_to_disk.fit_results_input)
        self.fit_light_curves.fit_results_output.connect(self.fits_to_disk.fit_results_input)

        if max_queue:  # Limit the number of light-curves fed into the pipeline
            self.simulate_light_curves.plasticc_data_input.maxsize = max_queue

        if sim_dir:
            self.sims_to_disk = SimulationToDisk(sim_dir)
            self.simulate_light_curves.simulation_output.connect(self.sims_to_disk.simulation_input)

        super(FittingPipeline, self).__init__()
