"""The ``fitting_pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). Here we demonstrate running a pipeline
synchronously.

.. code-block:: python

   >>> from snat_sim.fitting_pipeline import FittingPipeline

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
   >>> print('Simulation Processes:', pipeline.simulation_pool_size)
   >>> print('Fitting Processes:', pipeline.fitting_pool_size)
   >>> pipeline.run()

Module Docs
-----------
"""

import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import sncosmo

from . import plasticc, reference_stars, constants as const


class KillSignal:
    """Signal current process that it should try to exit gracefully"""

    pass


class ProcessManager:
    """Handles the starting and termination of forked processes"""

    def __init__(self, processes=tuple()):
        """Manage a collection of forked processes

        Args:
            processes (List[Process]): List of processes to manage
        """

        self._processes = processes

    def kill(self):
        """Kill all running pipeline processes without trying to exit gracefully"""

        for p in self._processes:
            p.terminate()

    def wait_for_exit(self):
        """Wait for the pipeline to finish running before continuing execution"""

        for p in self._processes:
            p.join()

    def run(self):
        """Similar to ``run_async`` but blocks further execution until finished"""

        self.run_async()
        self.wait_for_exit()

    def run_async(self):
        """Start all processes asynchronously"""

        for p in self._processes:
            p.start()


class OutputDataModel:
    """Enforces the data model of pipeline output files"""

    def __init__(self, sn_model):
        """Formats data from a given supernova model into a coherent tabular format

        Args:
            sn_model (Model): Supernova model to reflect in the outputted data model
        """

        self._sn_model = sn_model

    @staticmethod
    def build_table_entry(meta, fitted_model, result):
        """Combine light-curve fit results into single row matching the output table file format

        Args:
            meta          (dict): Meta data for the simulated light-curve
            fitted_model (Model): Supernova model fitted to the data
            result      (Result): sncosmo fit Result object

        Returns:
            A list of strings and floats
        """

        out_list = [meta['SNID']]
        out_list.extend(result.parameters)
        out_list.extend(result.errors.values())
        out_list.append(result.chisq)
        out_list.append(result.ndof)
        out_list.append(fitted_model.bandmag('bessellb', 'ab', time=fitted_model['t0']))
        out_list.append(fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo))
        out_list.append(result.message)
        return out_list

    def build_masked_entry(self, meta, excep):
        """Create a masked table entry for a failed light-curve fit

        Args:
            meta       (dict): Meta data for the simulated light-curve
            excep (Exception): Exception raised by the failed fit

        Returns:
            A list of strings and floats with masked values set as -99
        """

        num_columns = len(self.column_names)
        return [meta['SNID'], *np.full(num_columns - 2, -99), str(excep)]

    @property
    def column_names(self):
        """Return a list of column names for the data model

        Returns:
            List of column names as strings
        """

        col_names = ['SNID']
        col_names.extend(self._sn_model.param_names)
        col_names.extend(param + '_err' for param in self._sn_model.param_names)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')
        return col_names


class FittingPipeline(ProcessManager):
    """Pipeline of parallel processes for simulating and fitting light-curves"""

    def __init__(self, cadence, sim_model, fit_model, vparams, out_path,
                 quality_callback=None, max_queue=25, pool_size=None,
                 iter_lim=float('inf'), ref_stars=None, pwv_model=None):
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            cadence               (str): Cadence to use when simulating light-curves
            sim_model         (SNModel): Model to use when simulating light-curves
            fit_model         (SNModel): Model to use when fitting light-curves
            vparams         (list[str]): List of parameter names to vary in the fit
            out_path        (str, Path): Path to write results to (.csv extension is enforced)
            quality_callback (callable): Skip light-curves if this function returns False
            max_queue             (int): Maximum number of light-curves to store in pipeline at once
            pool_size             (int): Total number of workers to spawn. Defaults to CPU count
            iter_lim              (int): Limit number of processed light-curves (Useful for profiling)
            ref_stars       (List[str]): List of reference star types to calibrate simulated supernova with
            pwv_model        (PWVModel): Model for the PWV concentration the reference stars are observed at
        """

        self.pool_size = mp.cpu_count() if pool_size is None else pool_size
        if self.pool_size < 4:
            raise RuntimeError('Cannot spawn pipeline with less than 4 processes.')

        if (ref_stars is None) and not (pwv_model is None):
            raise ValueError('Cannot perform reference star subtraction without ``pwv_model`` argument')

        self.cadence = cadence
        self.sim_model = sim_model
        self.fit_model = fit_model
        self.out_path = Path(out_path).with_suffix('.csv')
        self.vparams = vparams
        self.quality_callback = quality_callback
        self.iter_lim = iter_lim
        self.reference_stars = ref_stars
        self.pwv_model = pwv_model

        self.out_path.parent.mkdir(exist_ok=True, parents=True)
        self.data_model = OutputDataModel(fit_model)

        # Set up queues to connect processes together
        manager = mp.Manager()
        self.queue_plasticc_lc = manager.Queue(max_queue // 2)
        self.queue_duplicated_lc = manager.Queue(max_queue // 2)
        self.queue_fit_results = manager.Queue()

        # Instantiate process objects - populates list self._processes
        super(FittingPipeline, self).__init__()
        self._init_processes()

    def _init_processes(self):
        """Instantiate forked processes but do not run them"""

        load_plasticc_process = mp.Process(target=self._load_queue_plasticc_lc)
        self._processes.append(load_plasticc_process)

        for _ in range(self.simulation_pool_size):
            duplicate_lc_process = mp.Process(target=self._duplicate_light_curves)
            self._processes.append(duplicate_lc_process)

        for _ in range(self.fitting_pool_size):
            fitting_process = mp.Process(target=self._fit_light_curves)
            self._processes.append(fitting_process)

        unload_results_process = mp.Process(target=self._unload_output_queue)
        self._processes.append(unload_results_process)

    @property
    def fitting_pool_size(self):
        """Number of processes used for fitting light-curves"""

        return (self.pool_size - 2) // 2

    @property
    def simulation_pool_size(self):
        """Number of processes used for simulating light-curves"""

        io_processes = 2
        return self.pool_size - io_processes - self.fitting_pool_size

    def _load_queue_plasticc_lc(self):
        """Load PLaSTICC light-curves from disk into the pipeline"""

        # Load light-curves into the first queue in the pipeline
        light_curve_iter = plasticc.iter_lc_for_cadence_model(self.cadence, model=11)
        for i, light_curve in enumerate(light_curve_iter):
            if i >= self.iter_lim:
                break

            self.queue_plasticc_lc.put(light_curve)

        # Signal the rest of the pipeline that there are no more light-curves
        # Load more than enough kill commands to make it through the pipeline
        for _ in range(self.pool_size):
            self.queue_plasticc_lc.put(KillSignal())

    def _duplicate_light_curves(self):
        """Simulate light-curves with atmospheric effects"""

        # Determine the redshift limit of the simulation model
        u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
        source_low = self.sim_model.source.minwave()
        z_limit = (u_band_low / source_low) - 1

        while not isinstance(light_curve := self.queue_plasticc_lc.get(), KillSignal):
            z = light_curve.meta['SIM_REDSHIFT_CMB']
            ra = light_curve.meta['RA']
            dec = light_curve.meta['DECL']

            # Skip the light-curve if it is outside the redshift range
            if z >= z_limit:
                continue

            # Simulate a duplicate light-curve with atmospheric effects
            duplicated_lc = plasticc.duplicate_plasticc_sncosmo(light_curve, self.sim_model, zp=30)

            if self.reference_stars is not None:
                pwv_los = self.pwv_model.pwv_los(duplicated_lc['time'], ra, dec, time_format='mjd')
                duplicated_lc = reference_stars.divide_ref_from_lc(duplicated_lc, pwv_los, self.reference_stars)

            # Skip if duplicated light-curve is not up to quality standards
            if self.quality_callback and not self.quality_callback(duplicated_lc):
                self.queue_fit_results.put(
                    self.data_model.build_masked_entry(light_curve.meta, ValueError('Failed quality check'))
                )
                continue

            self.queue_duplicated_lc.put(duplicated_lc)

        # Propagate kill signal
        self.queue_duplicated_lc.put(light_curve)

    def _fit_light_curves(self):
        """Fit light-curves"""

        warnings.simplefilter('ignore', category=DeprecationWarning)
        while not isinstance(light_curve := self.queue_duplicated_lc.get(), KillSignal):
            # Use the true light-curve parameters as the initial guess
            self.fit_model.update({k: v for k, v in light_curve.meta.items() if k in self.fit_model.param_names})

            try:
                result, fitted_model = sncosmo.fit_lc(
                    light_curve, self.fit_model, self.vparams,
                    guess_t0=False, guess_amplitude=False, guess_z=False, warn=False)

                self.queue_fit_results.put(self.data_model.build_table_entry(light_curve.meta, fitted_model, result))

            except Exception as excep:
                self.queue_fit_results.put(self.data_model.build_masked_entry(light_curve.meta, excep))

        # Propagate kill signal
        self.queue_fit_results.put(light_curve)

    def _unload_output_queue(self):
        """Retrieve fit results from the output queue and write results to file"""

        kill_count = 0  # Count closed upstream processes so this process knows when to exit

        with self.out_path.open('w') as outfile:
            outfile.write(','.join(self.data_model.column_names))

            while True:
                if isinstance(results := self.queue_fit_results.get(), KillSignal):
                    kill_count += 1
                    if kill_count >= self.fitting_pool_size:
                        # No more simulations or fits are being run
                        self._processes = []
                        return

                else:
                    new_line = ','.join(map(str, results)) + '\n'
                    outfile.write(new_line)
