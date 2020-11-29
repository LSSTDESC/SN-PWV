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

import inspect
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sncosmo
from astropy.cosmology import FlatwCDM
from iminuit import Minuit

from . import plasticc, reference_stars, constants as const


class KillSignal:
    """Signal current process that it should try to exit gracefully"""

    pass


class ProcessManager:
    """Handles the starting and termination of processes forked by the child class"""

    def __init__(self):
        self._processes = []

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

    def build_masked_entry(self, meta, fit_model, excep):
        """Create a masked table entry for a failed light-curve fit

        Args:
            meta       (dict): Meta data for the simulated light-curve
            fit_model (Model): Supernova model fitted to the data
            excep (Exception): Exception raised by the failed fit

        Returns:
            A list of strings and floats
        """

        num_columns = len(self.result_table_col_names(fit_model))
        return [meta['SNID'], *np.full(num_columns - 2, -99), str(excep)]

    @staticmethod
    def result_table_col_names(fit_model):
        """Return a list of column names for a given supernova model

        Args:
            fit_model (Model): Model with parameters to use as column names
        """

        col_names = ['SNID']
        col_names.extend(fit_model.param_names)
        col_names.extend(param + '_err' for param in fit_model.param_names)
        col_names.append('chisq')
        col_names.append('ndof')
        col_names.append('mb')
        col_names.append('abs_mag')
        col_names.append('message')
        return col_names


class FittingPipeline(ProcessManager, OutputDataModel):
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
        self.vparams = vparams
        self.quality_callback = quality_callback
        self.iter_lim = iter_lim
        self.reference_stars = ref_stars
        self.pwv_model = pwv_model

        self.out_path = Path(out_path).with_suffix('.csv')
        self.out_path.parent.mkdir(exist_ok=True, parents=True)

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
    def fitting_pool_size(self) -> int:
        """Number of processes used for fitting light-curves"""

        return (self.pool_size - 2) // 2

    @property
    def simulation_pool_size(self) -> int:
        """Number of processes used for simulating light-curves"""

        io_processes = 2
        return self.pool_size - io_processes - self.fitting_pool_size

    def _load_queue_plasticc_lc(self):
        """Load PLaSTICC light-curves from disk into the pipeline"""

        # The queue will block the for loop when it is full, limiting our memory usage
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

        # Determine redshift limit of the simulation model
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

                self.queue_fit_results.put(self.build_table_entry(light_curve.meta, fitted_model, result))

            except Exception as excep:
                self.queue_fit_results.put(self.build_masked_entry(light_curve.meta, self.fit_model, excep))

        # Propagate kill signal
        self.queue_fit_results.put(light_curve)

    def _unload_output_queue(self):
        """Retrieve fit results from the output queue and write results to file"""

        kill_count = 0  # Count closed upstream processes so this process knows when to exit

        with self.out_path.open('w') as outfile:
            outfile.write(','.join(self.result_table_col_names(self.fit_model)))

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


@pd.api.extensions.register_dataframe_accessor("snat_sim")
class CosmologyAccessor:
    """Chi-squared minimizer for fitting a cosmology to pipeline results"""

    def __init__(self, pandas_obj):
        self.data = pandas_obj

    def calc_distmod(self, abs_mag):
        """Return the distance modulus for an assumed absolute magnitude

        Args:
            abs_mag (float): The B-band absolute magnitude

        Returns:
            The distance modulus
        """

        return self.data['mb'] - abs_mag

    # noinspection PyPep8Naming
    def chisq(self, H0, Om0, abs_mag, w0, alpha, beta):
        """Calculate the chi-squared for given cosmological parameters

        Args:
            H0      (float): Hubble constant
            Om0     (float): Matter density
            abs_mag (float): SNe Ia intrinsic peak magnitude
            w0      (float): Dark matter equation of state
            alpha   (float): Stretch correction nuisance parameter
            beta    (float): Color correction nuisance parameter

        Returns:
            The chi-squared of the resulting cosmology
        """

        measured_mu = self.calc_distmod(abs_mag) + alpha * self.data['x1'] - beta * self.data['c']

        cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        modeled_mu = cosmology.distmod(self.data['z']).value
        return np.sum(((measured_mu - modeled_mu) ** 2) / (self.data['mb_err'] ** 2))

    # noinspection PyPep8Naming
    def chisq_grid(self, H0, Om0, abs_mag, w0, alpha, beta):
        """Calculate the chi-squared on a grid of cosmological parameters

        Arguments are automatically repeated along the grid so that the
        dimensions of each array match.

        Args:
            H0      (float, ndarray): Hubble constant
            Om0     (float, ndarray): Matter density
            abs_mag (float, ndarray): SNe Ia intrinsic peak magnitude
            w0      (float, ndarray): Dark matter equation of state
            alpha            (float): Stretch correction nuisance parameter
            beta             (float): Color correction nuisance parameter

        Returns:
            An array of chi-squared values
        """

        new_args = self._match_argument_dimensions(H0, Om0, abs_mag, w0, alpha, beta)
        return np.vectorize(self.chisq)(*new_args)

    @staticmethod
    def _match_argument_dimensions(*args):
        """Reshape arguments so they match the shape of the argument with the
        most dimensions.

        Args:
            *args (float, ndarray): Values to cast onto the grid

        Returns:
            A list with each argument cast to it's new shape
        """

        # Get the shape of the argument with the most dimensions
        grid_shape = np.shape(args[np.argmax([np.ndim(arg) for arg in args])])

        # Reshape each argument to match the dimensions from above
        return [np.full(grid_shape, arg) for arg in args]

    def minimize(self, **kwargs):
        """Fit cosmology to the instantiated data

        Kwargs:
            Accepts any iminuit style keyword arguments for parameters
              ``H0``, ``Om0``, ``abs_mag``, and ``w0``.

        Returns:
            Optimized Minuit object
        """

        minimizer = Minuit(self.chisq, **kwargs)
        minimizer.migrad()
        return minimizer

    def minimize_mc(self, samples, n=None, frac=None, statistic=None, **kwargs):
        """Fit cosmology to the instantiated data using monte carlo resampling

        Args:
            samples        (int): Number of samples to draw
            n              (int): Size of each sample. Cannot be used with ``frac``
            frac         (float): Fraction of data to include in each sample. Cannot be used with ``size``
            statistic (callable): Optionally apply a statistic to the returned values
            Accepts any iminuit style keyword arguments for parameters
              ``H0``, ``Om0``, ``abs_mag``, and ``w0``.

        Returns:
            List of optimized Minuit object or a dictionary of the applies statistic to those values
        """

        if statistic:
            samples = [self.data.sample(n=n, frac=frac).snat_sim.minimize(**kwargs).np_values() for _ in range(samples)]
            stat_val = statistic(samples)

            # Create a dictionary mapping the argument name to the applies statistic
            arg_names = inspect.getfullargspec(self.chisq).args
            samples = dict(zip(arg_names[1:], stat_val))  # First argument is self, so drop it

        else:
            samples = [self.data.sample(n=n, frac=frac).snat_sim.minimize(**kwargs) for _ in range(samples)]

        return samples
