from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import *
from typing import Dict, List

from egon.pipeline import Pipeline

from .lc_fitting import FitLightCurves, FitResultsToDisk
from .lc_simultion import SimulateLightCurves, SimulationToDisk
from .plasticc_io import LoadPlasticcSims
from ..models import SNModel
from ..reference_stars import VariableCatalog


class FittingPipeline(Pipeline):
    """Pipeline of parallel processes for simulating and fitting light-curves"""

    def __init__(
            self,
            cadence: str,
            sim_model: SNModel,
            fit_model: SNModel,
            vparams: List[str],
            out_path: Union[str, Path],
            sim_path: Union[str, Path] = None,
            fitting_pool: int = 1,
            simulation_pool: int = 1,
            bounds: Dict[str, Tuple[Number, Number]] = None,
            max_queue: int = 200,
            iter_lim: int = float('inf'),
            catalog: VariableCatalog = None,
            add_scatter=True
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            out_path: Path to write results to (.csv extension is enforced)
            sim_path: Optionally write simulated light-curves to disk in HDF5 format
            fitting_pool: Number of child processes allocated to simulating light-curves
            simulation_pool: Number of child processes allocated to fitting light-curves
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            catalog: Reference star catalog to calibrate simulated supernova with
            add_scatter: Add randomly generated scatter to simulated light-curve points
        """

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcSims(cadence, model=11, iter_lim=iter_lim)

        self.simulate_light_curves = SimulateLightCurves(
            sn_model=sim_model,
            catalog=catalog,
            num_processes=simulation_pool,
            include_pwv=sim_path is not None,
            add_scatter=add_scatter
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

        if sim_path:
            self.sims_to_disk = SimulationToDisk(sim_path)
            self.simulate_light_curves.simulation_output.connect(self.sims_to_disk.simulation_input)
