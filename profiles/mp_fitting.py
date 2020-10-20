# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Multiprocess script for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.

The processing pipeline is as follows:
  light-curve generator worker
  -> input Queue
  -> Pool of fitters
  -> output Queue
  -> write to file worker
"""

import multiprocessing as mp
import sys
from copy import copy
from itertools import repeat
from pathlib import Path
from typing import Union, Generator

import sncosmo
from astropy.table import Table

sys.path.insert(0, '../')
from snat_sim import modeling, plasticc, filters

model_type = Union[sncosmo.Model, modeling.Model]

filters.register_lsst_filters()

OUT_PATH = Path(__file__).resolve().parent / 'fit_results.csv'
CADENCE = 'alt_sched'


class FittingPipeline:
    """Parallelized implementation of light-curve fitting"""

    def __init__(self, light_curves: Generator, max_queue=25):
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            light_curves: Collection of light-curves to iterate over
        """

        # Defining maximum number of light-curves in each queue is
        # essential for memory management
        self.in_queue = mp.Queue(max_queue)
        self.out_queue = mp.Queue(max_queue)

        self.light_curves = light_curves
        self.keep_running = True

    def _load_input_queue(self) -> None:
        """Load light-curves into the inout queue for processing"""

        for lc in self.light_curves:
            self.in_queue.put(lc)

        self.keep_running = False

    def _fit_lc_wrapper(self, args):
        self._fit_lc_wrapper(*args)

    def _fit_light_curves(self, model: model_type, vparams: list, **kwargs) -> None:
        """Fit light-curves using the given model

        Light-curves are retrieved from the input queue, fitted, and then
        fit results are loaded into the output queue

        Args:
            model: Model to fit light-curves with
            vparams: List of parameter names to vary in the fit
            **kwargs: Any other arguments for ``sncosmo.fit_lc``
        """

        model = copy(model)

        while self.keep_running or not self.in_queue.empty():
            lc = self.in_queue.get()
            out_vals = list(lc.meta.values)

            # Use the true light-curve parameters as the initial guess
            lc.meta.pop('pwv', None)
            lc.meta.pop('res', None)

            # Fit the model without PWV
            model.update(lc.meta)
            _, fitted_model = sncosmo.fit_lc(lc, model, vparams, **kwargs)

            out_vals.extend(fitted_model.parameters)
            self.out_queue.put(out_vals)

    def _unload_output_queue(self) -> None:
        """Retrieve fit results from the output queue and write results to file"""

        with self.out_path.open() as outfile:
            while self.keep_running or not self.out_queue.empty():
                new_line = ','.join(self.out_queue.get()) + '\n'
                outfile.write(new_line)

    def run(self, out_path: Path, model: model_type, vparams: list) -> None:
        """Run fits of each light-curve and write results to file

        A ``.csv`` extension is enforced on the output file.

        Args:
            out_path: Path to write results to
            model: Model to fit light-curves with
            vparams: List of parameter names to vary in the fit
        """

        out_path.parent.mkdir(exist_ok=True, parents=True)
        self.out_path = out_path.with_suffix('.csv')

        load_data_process = mp.Process(target=self._load_input_queue)
        load_data_process.start()
        load_data_process.join()

        fitting_pool = mp.Pool()
        fitting_pool.starmap(self._fit_lc_wrapper, repeat((model, vparams)))
        fitting_pool.join()

        unload_results_process = mp.Process(target=self._unload_output_queue)
        unload_results_process.start()
        unload_results_process.join()

        load_data_process.close()
        fitting_pool.close()
        unload_results_process.close()


def iter_custom_lcs(
        model: model_type,
        cadence: str,
        iter_lim: int = None,
        gain: int = 20,
        skynr: int = 100,
        quality_callback: callable = None,
        verbose: bool = True):
    """Simulate light-curves for a given PLaSTICC cadence

    Args:
        model: Model to use in the simulations
        cadence: Cadence to use when simulating light-curves
        iter_lim: Stop iteration after given number of light-curves
        gain: Gain to use during simulation
        skynr: Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr
        quality_callback: Skip light-curves if this function returns False
        verbose: Display a progress bar
    """

    # Determine redshift limit of the given model
    u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
    source_low = model.source.minwave()
    zlim = (u_band_low / source_low) - 1

    counter = -1
    iter_lim = float('inf') if iter_lim is None else iter_lim
    for light_curve in plasticc.iter_lc_for_cadence_model(cadence, model=11, verbose=verbose):
        counter += 1
        if counter >= iter_lim:
            break

        if light_curve.meta['SIM_REDSHIFT_CMB'] >= zlim:
            continue

        model.set(ra=light_curve.meta['RA'], dec=light_curve.meta['DECL'])
        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(light_curve, model, gain=gain, skynr=skynr)

        if quality_callback and not quality_callback(duplicated_lc):
            continue

        yield duplicated_lc


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
    sn_model_with_pwv = modeling.Model(
        source='salt2-extended',
        effects=[variable_pwv_effect],
        effect_names=[''],
        effect_frames=['obs']
    )

    light_curve_generator = iter_custom_lcs(
        sn_model_with_pwv, cadence=CADENCE, iter_lim=100, quality_callback=passes_quality_cuts)

    FittingPipeline(light_curve_generator).run(
        out_path=OUT_PATH,
        model=sncosmo.Model('Salt2-extended'),
        vparams=['x0', 'x1', 'c'],
    )
