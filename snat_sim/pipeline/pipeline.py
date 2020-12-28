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

        # Define the nodes of the analysis pipeline
        self.load_plastic = LoadPlasticcSims(cadence=cadence, iter_lim=iter_lim)
        self.simulate_light_curves = SimulateLightCurves(
            sim_model=sim_model,
            ref_stars=ref_stars,
            pwv_model=pwv_model,
            quality_callback=quality_callback,
            num_processes=simulation_pool)

        data_model = DataModel(sim_model, fit_model)
        self.fit_light_curves = FitLightCurves(
            data_model=data_model, sn_model=fit_model, vparams=vparams, bounds=bounds, num_processes=fitting_pool)

        self.write_to_disk = FitResultsToDisk(data_model=data_model, out_path=out_path)

        # Connect pipeline nodes together
        self.load_plastic.lc_out.connect(self.simulate_light_curves.plasticc_data_input, maxsize=max_queue)
        self.simulate_light_curves.simulation_output.connect(self.fit_light_curves.light_curves_in)
        self.fit_light_curves.fit_results_out.connect(self.write_to_disk.fit_results_in)

        super(FittingPipeline, self).__init__()
