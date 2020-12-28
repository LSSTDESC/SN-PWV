"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). Here we demonstrate running a pipeline
synchronously.

.. code-block:: python

   >>> from snat_sim.pipeline import FittingPipeline

   >>> print('Instantiating pipeline...')
   >>> pipeline = FittingPipeline(
   >>>     cadence='alt_sched',
   >>>     sim_model=sn_model_sim,
   >>>     fit_model=sn_model_fit,
   >>>     vparams=['x0', 'x1', 'c'],
   >>>     out_path='./demo_out_path.csv',
   >>>     pool_size=6
   >>> )

   >>> print('I/O Processes: 2')
   >>> print('Simulation Processes:', pipeline.simulation_pool)
   >>> print('Fitting Processes:', pipeline.fitting_pool)
   >>> pipeline.run()

Module Docs
-----------
"""

from numbers import Number

from egon.pipeline import Pipeline

from nodes import *
from snat_sim.models import SNModel


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
            quality_callback: callable = None,
            max_queue: int = 50,
            iter_lim: int = float('inf'),
            ref_stars: Collection[str] = None,
            pwv_model: SNModel = None
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
            quality_callback: Skip light-curves if this function returns False
            max_queue: Maximum number of light-curves to store in pipeline at once
            iter_lim: Limit number of processed light-curves (Useful for profiling)
            ref_stars: List of reference star types to calibrate simulated supernova with
            pwv_model: Model for the PWV concentration the reference stars are observed at
        """

        if (ref_stars is None) and not (pwv_model is None):
            raise ValueError('Cannot perform reference star subtraction without ``pwv_model`` argument')

        data_model = DataModel(sim_model, fit_model)

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcSims(cadence=cadence, iter_lim=iter_lim)
        self.simulate_light_curves = SimulateLightCurves(
            data_model=data_model,
            sn_model=sim_model,
            ref_stars=ref_stars,
            pwv_model=pwv_model,
            quality_callback=quality_callback,
            num_processes=simulation_pool)

        self.fit_light_curves = FitLightCurves(
            data_model=data_model, sn_model=fit_model, vparams=vparams, bounds=bounds, num_processes=fitting_pool)

        self.write_to_disk = FitResultsToDisk(data_model=data_model, out_path=out_path)

        # Connect pipeline nodes together
        self.load_plastic.lc_output.connect(self.simulate_light_curves.plasticc_data_input, maxsize=max_queue)
        self.simulate_light_curves.simulation_output.connect(self.fit_light_curves.light_curves_input)
        self.simulate_light_curves.masked_failure_output.connect(self.write_to_disk.fit_results_input)
        self.fit_light_curves.fit_results_output.connect(self.write_to_disk.fit_results_input)

        super(FittingPipeline, self).__init__()
