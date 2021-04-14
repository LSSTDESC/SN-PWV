"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

SubModules
----------

Although the ``pipeline`` module provides a prebuilt data analysis pipeline,
you can also build customized pipelines using any of the included nodes.
Relevant documentation can be found in the following pages:

.. autosummary::
   :nosignatures:

   data_model
   nodes

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

from numbers import Number
from pathlib import Path
from typing import *
from typing import Dict, List

from egon.pipeline import Pipeline

from . import nodes
from .nodes import FitLightCurves, LoadPlasticcCadence, SimulateLightCurves, WritePipelinePacket
from ..models import SNModel, VariableCatalog
from ..plasticc import PLaSTICC


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
            catalog: VariableCatalog = None,
            add_scatter: bool = True,
            fixed_snr: Optional[float] = None,
            overwrite: bool = False
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            out_path: Path to write results to
            fitting_pool: Number of child processes allocated to simulating light-curves
            simulation_pool: Number of child processes allocated to fitting light-curves
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            catalog: Reference star catalog to calibrate simulated supernova with
            add_scatter: Add randomly generated scatter to simulated light-curve points
            fixed_snr: Simulate light-curves with a fixed signal to noise ratio
            overwrite: Whether to overwrite an existing output file
        """

        out_path = Path(out_path)
        if (not overwrite) and out_path.exists():
            raise FileExistsError(f'Cannot overwrite existing results: {out_path}')

        out_path.parent.mkdir(exist_ok=True)

        # Define the nodes of the analysis pipeline
        cadence = PLaSTICC(cadence, model=11)
        self.load_plastic = LoadPlasticcCadence(cadence, iter_lim=iter_lim)
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
        self.load_plastic.output.connect(self.simulate_light_curves.input)
        self.simulate_light_curves.success_output.connect(self.fit_light_curves.input)
        self.simulate_light_curves.failure_output.connect(self.write_to_disk.input)
        self.fit_light_curves.success_output.connect(self.write_to_disk.input)
        self.fit_light_curves.failure_output.connect(self.write_to_disk.input)

        if max_queue:  # Limit the number of light-curves fed into the pipeline
            self.simulate_light_curves.input.maxsize = max_queue
            self.fit_light_curves.input.maxsize = max_queue * simulation_pool
            self.write_to_disk.input.maxsize = max_queue * simulation_pool
