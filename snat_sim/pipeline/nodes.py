import warnings
from pathlib import Path
from typing import *

import sncosmo
from egon.connectors import Output, Input
from egon.nodes import Source, Node, Target

from .data_model import DataModel
from .. import plasticc, reference_stars
from ..models import SNModel, PWVModel


class LoadPlasticcSims(Source):
    lc_out = Output()

    def __init__(self, cadence: str, iter_lim: int = float('inf'), num_processes: int = 1) -> None:
        """Source node for loading PLaSTICC light-curves from disk

        Args:
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

            self.lc_out.put(light_curve)


class SimulateLightCurves(Node):
    plasticc_data_input = Input()
    simulation_output = Output()

    def __init__(
            self,
            sim_model: SNModel,
            ref_stars: Collection[str],
            pwv_model: PWVModel,
            quality_callback: callable = None,
            num_processes: int = 1
    ) -> None:

        super().__init__(num_processes)
        self.sim_model = sim_model
        self.ref_stars = ref_stars
        self.pwv_model = pwv_model
        self.quality_callback = quality_callback

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
            duplicated_lc = plasticc.duplicate_plasticc_sncosmo(light_curve, self.sim_model, zp=30)

            if self.ref_stars is not None:
                pwv_los = self.pwv_model.pwv_los(duplicated_lc['time'], ra, dec, time_format='mjd')
                duplicated_lc = reference_stars.divide_ref_from_lc(duplicated_lc, pwv_los, self.ref_stars)

            # Skip if duplicated light-curve is not up to quality standards
            if self.quality_callback and not self.quality_callback(duplicated_lc):
                self.queue_fit_results.put(
                    self.data_model.build_masked_entry(duplicated_lc.meta, ValueError('Failed quality check'))
                )
                continue

            self.simulation_output.put(duplicated_lc)


class FitLightCurves(Node):
    light_curves_in = Input()
    fit_results_out = Output()

    def __init__(
            self,
            data_model: DataModel,
            sn_model: SNModel,
            vparams: List[str],
            bounds: Dict = None,
            num_processes: int = 1
    ) -> None:
        super(FitLightCurves, self).__init__(num_processes)
        self.data_model = data_model
        self.fit_model = sn_model
        self.vparams = vparams
        self.bounds = bounds

    def action(self) -> None:
        """Fit light-curves"""

        warnings.simplefilter('ignore', category=DeprecationWarning)
        for light_curve in self.light_curves_in.iter_get():
            # Use the true light-curve parameters as the initial guess
            self.fit_model.update({k: v for k, v in light_curve.meta.items() if k in self.fit_model.param_names})

            try:
                result, fitted_model = sncosmo.fit_lc(
                    light_curve, self.fit_model, self.vparams, bounds=self.bounds,
                    guess_t0=False, guess_amplitude=False, guess_z=False, warn=False)

                self.fit_results_out.put(self.data_model.build_table_entry(light_curve.meta, fitted_model, result))

            except Exception as excep:
                self.fit_results_out.put(self.data_model.build_masked_entry(light_curve.meta, excep))


class FitResultsToDisk(Target):
    fit_results_in = Input()

    def __init__(self, data_model: DataModel, out_path: Union[str, Path], num_processes: int = 1) -> None:
        super(FitResultsToDisk, self).__init__(num_processes)
        self.data_model = data_model
        self.out_path = out_path

    def setup(self) -> None:
        self.out_path.parent.mkdir(exist_ok=True, parents=True)

    def action(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open('w') as outfile:
            outfile.write(','.join(self.data_model.column_names))

            for results in self.fit_results_in.iter_get():
                new_line = ','.join(map(str, results)) + '\n'
                outfile.write(new_line)
