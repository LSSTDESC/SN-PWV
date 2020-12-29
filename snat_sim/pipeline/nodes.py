"""The ``nodes`` module defines the individual data processing nodes for the
analysis pipeline.
"""

import warnings
from copy import copy
from pathlib import Path
from typing import *

import sncosmo
from astropy.table import Table
from egon.connectors import Output, Input
from egon.nodes import Source, Node, Target

from .data_model import DataModel
from .. import plasticc, reference_stars, constants as const
from ..models import SNModel, PWVModel, ObservedCadence


class LoadPlasticcSims(Source):
    """Pipeline node for loading PLaSTICC data from disk

    Connectors:
        lc_output: The loaded PLaSTICC light-curves as ``astropy.Table`` objects
    """

    lc_output = Output()

    def __init__(self, cadence: str, iter_lim: int = float('inf'), num_processes: int = 1) -> None:
        """Source node for loading PLaSTICC light-curves from disk

        Args:
            cadence: Cadence to use when simulating light-curves
            iter_lim: Exit after loading the given number of light-curves
            num_processes: Number of processes to allocate to the node
        """

        super(LoadPlasticcSims, self).__init__(num_processes)
        self.cadence = cadence
        self.iter_lim = iter_lim

    def action(self) -> None:
        """Load PLaSTICC light-curves from disk"""

        light_curve_iter = plasticc.iter_lc_for_cadence_model(self.cadence, model=11)
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

    plasticc_data_input = Input(maxsize=100)
    simulation_output = Output()
    masked_failure_output = Output()

    def __init__(
            self,
            data_model: DataModel,
            sn_model: SNModel,
            ref_stars: Collection[str],
            pwv_model: PWVModel,
            quality_callback: callable = None,
            num_processes: int = 1,
            abs_mb: float = const.betoule_abs_mb,
            cosmo=const.betoule_cosmo
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when simulating light-curves
            ref_stars: List of reference star types to calibrate simulated supernova with
            pwv_model: Model for the PWV concentration the reference stars are observed at
            quality_callback: Skip light-curves if this function returns False
            num_processes: Number of processes to allocate to the node
            abs_mb: The absolute B-band magnitude of the simulated SNe
            cosmo: Cosmology to assume in the simulation
        """

        super().__init__(num_processes)
        self.data_model = data_model
        self.sim_model = sn_model
        self.ref_stars = ref_stars
        self.pwv_model = pwv_model
        self.quality_callback = quality_callback
        self.abs_mb = abs_mb
        self.cosmo = cosmo

    def duplicate_plasticc_lc(self, plasticc_lc: Table, zp: float = 30) -> Table:
        """Duplicate a plastic light-curve using the simulation model

        Args:
            plasticc_lc: The light-curve to duplicate
            zp: Zero-point of the duplicated light-curve
        """

        params, plasticc_cadence = ObservedCadence.from_plasticc(plasticc_lc, zp=zp)

        model_for_sim = copy(self.sim_model)
        model_for_sim.update({p: v for p, v in params.items() if p in model_for_sim.param_names})
        model_for_sim.set_source_peakabsmag(self.abs_mb, 'standard::b', 'AB', cosmo=self.cosmo)
        duplicated = model_for_sim.simulate_lc(plasticc_cadence)
        duplicated.meta = params
        return duplicated

    def action(self) -> None:
        """Simulate light-curves with atmospheric effects"""

        # Determine the redshift limit of the simulation model
        u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
        source_low = self.sim_model.source.minwave()
        z_limit = (u_band_low / source_low) - 1

        for light_curve in self.plasticc_data_input.iter_get():
            z = light_curve.meta['SIM_REDSHIFT_CMB']
            ra = light_curve.meta['RA']
            dec = light_curve.meta['DECL']

            # Skip the light-curve if it is outside the redshift range
            if z >= z_limit:
                continue

            # Simulate a duplicate light-curve with atmospheric effects
            duplicated_lc = self.duplicate_plasticc_lc(light_curve, zp=30)

            if self.ref_stars is not None:
                pwv_los = self.pwv_model.pwv_los(duplicated_lc['time'], ra, dec, time_format='mjd')
                duplicated_lc = reference_stars.divide_ref_from_lc(duplicated_lc, pwv_los, self.ref_stars)

            # Skip if duplicated light-curve is not up to quality standards
            if self.quality_callback and not self.quality_callback(duplicated_lc):
                self.masked_failure_output.put(
                    self.data_model.build_masked_entry(duplicated_lc.meta, ValueError('Failed quality check'))
                )
                continue

            self.simulation_output.put(duplicated_lc)


class FitLightCurves(Node):
    """Pipeline node for fitting simulated light-curves

    Connectors:
        light_curves_input: Light-curves to fit
        fit_results_output: Fit results as a list
    """

    light_curves_input = Input()
    fit_results_output = Output()

    def __init__(
            self,
            data_model: DataModel,
            sn_model: SNModel,
            vparams: List[str],
            bounds: Dict = None,
            num_processes: int = 1
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            data_model: Data model for formatting output data
            sn_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            num_processes: Number of processes to allocate to the node
        """

        super(FitLightCurves, self).__init__(num_processes)
        self.data_model = data_model
        self.fit_model = sn_model
        self.vparams = vparams
        self.bounds = bounds

    def action(self) -> None:
        """Fit light-curves"""

        warnings.simplefilter('ignore', category=DeprecationWarning)
        for light_curve in self.light_curves_input.iter_get():
            # Use the true light-curve parameters as the initial guess
            self.fit_model.update({k: v for k, v in light_curve.meta.items() if k in self.fit_model.param_names})

            try:
                result, fitted_model = sncosmo.fit_lc(
                    light_curve, self.fit_model, self.vparams, bounds=self.bounds,
                    guess_t0=False, guess_amplitude=False, guess_z=False, warn=False)

                self.fit_results_output.put(self.data_model.build_table_entry(light_curve.meta, fitted_model, result))

            except Exception as excep:
                self.fit_results_output.put(self.data_model.build_masked_entry(light_curve.meta, excep))


class FitResultsToDisk(Target):
    """Pipeline node for writing fit results to disk

    Connectors:
        fit_results_input: List of values to write as single line in CSV format
    """

    fit_results_input = Input()

    def __init__(self, data_model: DataModel, out_path: Union[str, Path], num_processes: int = 1) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            data_model: Data model for formatting output data
            out_path: Path to write results to (.csv extension is enforced)
        """

        super(FitResultsToDisk, self).__init__(num_processes)
        self.data_model = data_model
        self.out_path = out_path

    def setup(self) -> None:
        """Ensure the parent directory of the destination file exists"""

        self.out_path.parent.mkdir(exist_ok=True, parents=True)

    def action(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open('w') as outfile:
            outfile.write(','.join(self.data_model.column_names))
            for results in self.fit_results_input.iter_get():
                new_line = ','.join(map(str, results)) + '\n'
                outfile.write(new_line)
