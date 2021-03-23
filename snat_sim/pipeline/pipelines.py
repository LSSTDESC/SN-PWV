from __future__ import annotations

import warnings
from numbers import Number
from pathlib import Path
from typing import *
from typing import Dict, List

from egon.pipeline import Pipeline
from tables import NaturalNameWarning

from snat_sim.pipeline.nodes import FitLightCurves, SimulateLightCurves, WritePipelinePacket
from snat_sim.pipeline.nodes import LoadPlasticcCadence
from ..models import SNModel
from ..reference_stars import VariableCatalog

warnings.filterwarnings('ignore', category=NaturalNameWarning)


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
            add_scatter: bool = True,
            fixed_snr: Optional[float] = None
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            out_path: Path to write results to
            sim_path: Optionally write simulated light-curves to disk
            fitting_pool: Number of child processes allocated to simulating light-curves
            simulation_pool: Number of child processes allocated to fitting light-curves
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            catalog: Reference star catalog to calibrate simulated supernova with
            add_scatter: Add randomly generated scatter to simulated light-curve points
        """

        if out_path.exists() or (sim_path and sim_path.exists()):
            raise FileExistsError(f'Cannot overwrite existing results: {out_path}')

        out_path.parent.mkdir(exist_ok=True)

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcCadence(cadence, model=11, iter_lim=iter_lim)
        self.write_to_disk = WritePipelinePacket(out_path)

        self.simulate_light_curves = SimulateLightCurves(
            sn_model=sim_model,
            catalog=catalog,
            num_processes=simulation_pool,
            add_scatter=add_scatter,
            fixed_snr=fixed_snr
        )

        self.fit_light_curves = FitLightCurves(
            sn_model=fit_model,
            vparams=vparams,
            bounds=bounds,
            num_processes=fitting_pool)

        # Connect pipeline nodes together
        self.load_plastic.output.connect(self.simulate_light_curves.cadence_data_input)
        self.simulate_light_curves.success_output.connect(self.fit_light_curves.light_curves_input)
        self.simulate_light_curves.failure_output.connect(self.write_to_disk.data_input)
        self.fit_light_curves.success_output.connect(self.write_to_disk.data_input)
        self.fit_light_curves.failure_output.connect(self.write_to_disk.data_input)

        if max_queue:  # Limit the number of light-curves fed into the pipeline
            self.simulate_light_curves.cadence_data_input.maxsize = max_queue
