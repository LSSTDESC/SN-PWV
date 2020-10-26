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
import warnings
from copy import copy
from pathlib import Path
from typing import Union

import sncosmo
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snat_sim import modeling, plasticc, filters

model_type = Union[sncosmo.Model, modeling.Model]
filters.register_lsst_filters()

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
            max_queue=25,
            pool_size: int = None):
        """Fit light-curves using multiple processes and combine results into an output file

        The ``max_queue`` argument can be used to limit **duplicate**
        memory usage by restricting the number of light-curves that are read
        into the  pipeline at once. However, it does not effect memory usage
        by the underlying file parser. In general increasing the pool size
        has minimal performance impact.

        Args:
            cadence: Cadence to use when simulating light-curves
            sim_model: Model to use when simulating light-curves
            fit_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            gain: Gain to use during simulation
            skynr: Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr
            quality_callback: Skip light-curves if this function returns False
            max_queue: Maximum number of light-curves to store in memory at once
            pool_size: Total number of workers to spawn. Defaults to CPU count
        """

        self.pool_size = mp.cpu_count() if pool_size is None else pool_size
        if self.pool_size < 4:
            raise RuntimeError('Cannot spawn pipeline with less than 4 processes.')

        self.cadence = cadence
        self.sim_model = sim_model
        self.fit_model = fit_model
        self.vparams = vparams
        self.gain = gain
        self.skynr = skynr
        self.quality_callback = quality_callback
        self.out_path = None  # To be set when ``run`` is called

        manager = mp.Manager()
        self.queue_plasticc_lc = manager.Queue(max_queue // 2)
        self.queue_duplicated_lc = manager.Queue(max_queue // 2)
        self.queue_fit_results = manager.Queue()

    @property
    def fitting_pool_size(self) -> int:
        """Number of processes used for fitting light-curves"""

        return (self.pool_size - 2) // 2

    @property
    def simulation_pool_size(self) -> int:
        """Number of processes used for simulating light-curves"""

        return self.pool_size - self.fitting_pool_size

    def _load_queue_plasticc_lc(self) -> None:
        """Load light-curves from a given PLaSTICC cadence into a the pipeline"""

        # The queue will block the for loop when it is full, limiting our memory usage
        for light_curve in plasticc.iter_lc_for_cadence_model(self.cadence, model=11, verbose=True):
            self.queue_plasticc_lc.put(light_curve)

        # Signal the rest of the pipeline that there are no more light-curves
        # Load more than enough kill commands to make it through the pipeline
        for _ in range(self.pool_size):
            self.queue_plasticc_lc.put('KILL')

    def _duplicate_light_curves(self) -> None:
        """Simulate light-curves for a given PLaSTICC cadence with atmospheric effects"""

        # Determine redshift limit of the simulation model
        u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
        source_low = self.sim_model.source.minwave()
        z_limit = (u_band_low / source_low) - 1

        while light_curve := self.queue_plasticc_lc.get() != 'KILL':

            # Skip the light-curve if it is outside the redshift range
            if light_curve.meta['SIM_REDSHIFT_CMB'] >= z_limit:
                continue

            # Simulate a duplicate light-curve with atmospheric effects
            self.sim_model.set(ra=light_curve.meta['RA'], dec=light_curve.meta['DECL'])
            duplicated_lc = plasticc.duplicate_plasticc_sncosmo(
                light_curve, self.sim_model, gain=self.gain, skynr=self.skynr)

            # Skip if duplicated light-curve is not up to quality standards
            if self.quality_callback and not self.quality_callback(duplicated_lc):
                continue

            self.queue_duplicated_lc.put(duplicated_lc)

        self.queue_duplicated_lc.put('KILL')

    def _fit_light_curves(self) -> None:
        """Fit light-curves using the given model"""

        fit_model = copy(self.fit_model)
        while light_curve := self.queue_duplicated_lc.get() != 'KILL':
            out_vals = list(light_curve.meta.values())

            # Use the true light-curve parameters as the initial guess
            fit_model.update({k: v for k, v in light_curve.meta.items() if k in fit_model.param_names})

            # Fit the model without PWV
            _, fitted_model = sncosmo.fit_lc(light_curve, fit_model, self.vparams)

            out_vals.extend(fitted_model.parameters)
            self.queue_fit_results.put(out_vals)

        self.queue_fit_results.put('KILL')

    def _unload_output_queue(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        kill_count = 0
        with self.out_path.open('w') as outfile:
            while True:
                if (results := self.queue_fit_results.get()) == 'KILL':
                    kill_count += 1
                    if kill_count >= self.fitting_pool_size:
                        return  # No more fits are being run

                new_line = ','.join(map(str, results)) + '\n'
                outfile.write(new_line)

    def run(self, out_path: Path) -> None:
        """Run fits of each light-curve and write results to file

        A ``.csv`` extension is enforced on the output file.

        Args:
            out_path: Path to write results to
        """

        out_path.parent.mkdir(exist_ok=True, parents=True)
        self.out_path = out_path.with_suffix('.csv')

        processes = []  # Accumulator for processes so they can be joined at the end

        load_plasticc_process = mp.Process(target=self._load_queue_plasticc_lc)
        load_plasticc_process.start()
        processes.append(load_plasticc_process)

        for _ in range(self.simulation_pool_size):
            duplicate_lc_process = mp.Process(target=self._duplicate_light_curves)
            duplicate_lc_process.start()
            processes.append(duplicate_lc_process)

        for _ in range(self.fitting_pool_size):
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

    available_cadences = plasticc.get_available_cadences()
    cadence_name = sys.argv[1]
    if cadence_name not in available_cadences:
        raise ValueError(f'Cadence {cadence_name} not available from local cadences: {available_cadences}')

    # Characterize the atmospheric variability
    # Set PWV to a constant while developing
    pwv_interpolator = lambda *args: 5
    variable_pwv_effect = modeling.VariablePWVTrans(pwv_interpolator)
    variable_pwv_effect.set(res=5)

    # Build models with and without atmospheric effects
    model_without_pwv = sncosmo.Model('Salt2-extended')
    model_with_pwv = modeling.Model(
        source='salt2-extended',
        effects=[variable_pwv_effect],
        effect_names=[''],
        effect_frames=['obs']
    )

    output = Path(__file__).resolve().parent.parent / 'results' / 'fit_results.csv'
    FittingPipeline(
        cadence=cadence_name,
        sim_model=model_with_pwv,
        fit_model=model_without_pwv,
        vparams=['x0', 'x1', 'c'],
        pool_size=10
    ).run(out_path=output)
