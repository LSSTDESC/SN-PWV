# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Multiprocess script for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.

The processing pipeline is as follows:
  x- WORKER: Load plasticc light-curves from disk -> Queue
  -> POOL: Retrieve Plastic light-curves and add atmospheric effects -> Queue
  -> POOL: Fit duplicated light-curves and determine fitted parameters -> Queue
  -x WORKER: Write fitted parameters to file
"""

import multiprocessing as mp
import sys
from copy import copy
from pathlib import Path
from typing import Union

import sncosmo
from astropy.table import Table

sys.path.insert(0, '../')
from snat_sim import modeling, plasticc, filters

model_type = Union[sncosmo.Model, modeling.Model]
filters.register_lsst_filters()

OUT_PATH = Path(__file__).resolve().parent / 'fit_results.csv'
CADENCE = 'alt_sched'  # Todo: Make this a command line argument


class FittingPipeline:
    def __init__(
            self,
            cadence: str,
            sim_model: model_type,
            fit_model: model_type,
            vparams: list,
            gain: int = 20,
            skynr: int = 100,
            quality_callback: callable = None,
            max_queue=25):
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            gain: Gain to use during simulation
            skynr: Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr
            quality_callback: Skip light-curves if this function returns False
            max_queue: Maximum number of light-curves to store in memory at once
        """

        self.cadence = cadence
        self.sim_model = sim_model
        self.fit_model = fit_model
        self.vparams = vparams
        self.gain = gain
        self.skynr = skynr
        self.quality_callback = quality_callback

        manager = mp.Manager()
        self.queue_plasticc_lc = manager.Queue(max_queue // 2)
        self.queue_duplicated_lc = manager.Queue(max_queue // 2)
        self.queue_fit_results = manager.Queue()
        self.keep_running = True
        self.out_path = None  # To be set when ``run`` is called

    def _load_queue_plasticc_lc(self) -> None:
        """Load light-curves from a given PLaSTICC cadence"""

        # The queue will block the for loop when it is fool, limiting our memory usage
        for light_curve in plasticc.iter_lc_for_cadence_model(self.cadence, model=11, verbose=True):
            print('reading', light_curve.meta['SNID'])

            self.queue_plasticc_lc.put(light_curve)

    def _duplicate_light_curves(self) -> None:
        """Simulate light-curves for a given PLaSTICC cadence"""

        # Determine redshift limit of the given model
        u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
        source_low = self.sim_model.source.minwave()
        z_limit = (u_band_low / source_low) - 1

        # Skip the light-curve if it is outside the redshift range
        light_curve = self.queue_plasticc_lc.get()
        print('duplicating', light_curve.meta['SNID'])

        if light_curve.meta['SIM_REDSHIFT_CMB'] >= z_limit:
            print('dropping', light_curve.meta['SNID'])
            return

        # Simulate a duplicate light-curve with atmospheric effects
        self.sim_model.set(ra=light_curve.meta['RA'], dec=light_curve.meta['DECL'])
        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(
            light_curve, self.sim_model, gain=self.gain, skynr=self.skynr)

        # Skip if duplicated light-curve is not up to quality standards
        if self.quality_callback and not self.quality_callback(duplicated_lc):
            print('dropping', light_curve.meta['SNID'])
            return

        self.queue_duplicated_lc.put(duplicated_lc)

    def _fit_light_curves(self) -> None:
        """Fit light-curves using the given model"""

        fit_model = copy(self.fit_model)

        while self.keep_running or not self.queue_duplicated_lc.empty():
            lc = self.queue_duplicated_lc.get()
            print('fitting', lc.meta['SNID'])
            out_vals = list(lc.meta.values)

            # Use the true light-curve parameters as the initial guess
            lc.meta.pop('pwv', None)
            lc.meta.pop('res', None)

            # Fit the model without PWV
            fit_model.update(lc.meta)
            _, fitted_model = sncosmo.fit_lc(lc, fit_model, self.vparams)

            out_vals.extend(fitted_model.parameters)
            self.queue_fit_results.put(out_vals)

    def _unload_output_queue(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open('w') as outfile:
            while self.keep_running or not self.queue_fit_results.empty():
                new_line = ','.join(self.queue_fit_results.get()) + '\n'
                outfile.write(new_line)

    def run(self, out_path: Path, pool_size: int = None) -> None:
        """Run fits of each light-curve and write results to file

        A ``.csv`` extension is enforced on the output file.

        Args:
            out_path: Path to write results to
            pool_size: Total number of workers to spawn. Defaults to CPU count
        """

        pool_size = mp.cpu_count() if pool_size is None else pool_size
        if pool_size < 4:
            raise RuntimeError('Cannot spawn multiprocessing with less than 4 processes.')

        out_path.parent.mkdir(exist_ok=True, parents=True)
        self.out_path = out_path.with_suffix('.csv')

        # Collect processes so they can be joined at the end
        processes = []

        # Save two processes for reading / writing to disk. All others
        # Used for simulation / fitting
        processes_available_for_pools = pool_size - 2
        simulation_pool_size = processes_available_for_pools // 2
        fitting_pool_size = pool_size - simulation_pool_size

        load_plasticc_process = mp.Process(target=self._load_queue_plasticc_lc)
        load_plasticc_process.start()
        processes.append(load_plasticc_process)

        for _ in range(simulation_pool_size):
            duplicate_lc_process = mp.Process(target=self._duplicate_light_curves)
            duplicate_lc_process.start()
            processes.append(duplicate_lc_process)

        for _ in range(fitting_pool_size):
            fitting_process = mp.Process(target=self._fit_light_curves)
            fitting_process.start()
            processes.append(fitting_process)

        unload_results_process = mp.Process(target=self._unload_output_queue)
        unload_results_process.start()
        processes.append(unload_results_process)

        for p in processes:
            p.join()


def passes_quality_cuts(light_curve: Table) -> bool:
    """Return whether light-curve has 2+ two bands each with 1+ data point with SNR > 5

    Args:
        light_curve: Astropy table with sncosmo formatted light-curve data

    Returns:
        A boolean
    """

    if light_curve.meta['z'] > .88:
        return False

    light_curve = light_curve.group_by('band')

    passed_cuts = []
    for band_lc in light_curve.groups:
        passed_cuts.append((band_lc['flux'] / band_lc['fluxerr'] > 5).any())

    return sum(passed_cuts) >= 2


if __name__ == '__main__':
    # Characterize the atmospheric variability
    pwv_interpolator = lambda *args: 5
    variable_pwv_effect = modeling.VariablePWVTrans(pwv_interpolator)
    variable_pwv_effect.set(res=5)

    # Build a model with atmospheric effects
    model_with_pwv = modeling.Model(
        source='salt2-extended',
        effects=[variable_pwv_effect],
        effect_names=[''],
        effect_frames=['obs']
    )

    model_without_pwv = sncosmo.Model('Salt2-extended')

    FittingPipeline(
        cadence='alt_sched',
        sim_model=model_with_pwv,
        fit_model=model_without_pwv,
        vparams=['x0', 'x1', 'c'],
    ).run(out_path=OUT_PATH)
