"""Defines the individual data processing nodes used to construct complete
data analysis pipelines.

.. note:: Nodes are built on the ``egon`` framework. For more information see the
   official `Egon Documentation <https://mwvgroup.github.io/Egon/>`_.

Module Docs
-----------
"""

from __future__ import annotations

import warnings
from copy import copy
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from astropy.cosmology.core import Cosmology
from egon.connectors import Input, Output
from egon.nodes import Node, Source, Target

from .. import constants as const
from ..models import LightCurve, ObservedCadence, SNFitResult, SNModel, VariableCatalog
from ..pipeline.data_model import PipelinePacket
from ..plasticc import PLAsTICC


class LoadPlasticcCadence(Source):
    """Pipeline node for loading PLAsTICC cadence data from disk

    Connectors:
        output: Emits a pipeline packet decorated with the snid, simulation parameters, and cadence
    """

    def __init__(
            self,
            plasticc_dao: PLAsTICC,
            iter_lim: int = float('inf'),
            override_zp: float = 30,
            verbose: bool = True,
            num_processes: int = 1
    ) -> None:
        """Source node for loading PLAsTICC cadence data from disk

        This node can only be run using a single process. This can be the main
        process (``num_processes=0``) or a single forked process (``num_processes=1``.)

        Args:
            plasticc_dao: A PLAsTICC data access object
            iter_lim: Exit after loading the given number of light-curves
            override_zp: Overwrite the zero-point used by plasticc with this number
            verbose: Display a progress bar
            num_processes: Number of processes to allocate to the node (must be 0 or 1 for this node)
        """

        if num_processes not in (0, 1):
            raise RuntimeError('Number of processes for ``LoadPlasticcCadence`` must be 0 or 1.')

        self.cadence = plasticc_dao
        self.iter_lim = iter_lim
        self.override_zp = override_zp
        self.verbose = verbose

        # Node connectors
        self.output = Output('Loading Cadence Output')
        super().__init__(num_processes=num_processes)

    def action(self) -> None:
        """Load PLAsTICC cadence data from disk"""

        for snid, params, cadence in self.cadence.iter_cadence(iter_lim=self.iter_lim, verbose=self.verbose):
            cadence.zp = np.full_like(cadence.zp, self.override_zp)
            self.output.put(PipelinePacket(snid, params, cadence))


class SimulateLightCurves(Node):
    """Pipeline node for simulating light-curves based on PLAsTICC cadences

    Connectors:
        input: A Pipeline Packet
        success_output: Emits pipeline packets successfully decorated with a simulated light-curve
        failure_output: Emits pipeline packets for cases where the simulation procedure failed
    """

    def __init__(
            self,
            sn_model: SNModel,
            catalog: VariableCatalog = None,
            num_processes: int = 1,
            add_scatter: bool = True,
            fixed_snr: Optional[float] = None,
            abs_mb: float = const.betoule_abs_mb,
            cosmo: Cosmology = const.betoule_cosmo
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when simulating light-curves
            catalog: Optional reference start catalog to calibrate simulated flux values to
            num_processes: Number of processes to allocate to the node
            abs_mb: The absolute B-band magnitude of the simulated SNe
            cosmo: Cosmology to assume in the simulation
        """

        self.sim_model = copy(sn_model)
        self.catalog = catalog
        self.add_scatter = add_scatter
        self.fixed_snr = fixed_snr
        self.abs_mb = abs_mb
        self.cosmo = cosmo

        # Node connectors
        self.input = Input('Simulated Cadence')
        self.success_output = Output('Simulation Success')
        self.failure_output = Output('Simulation Failure')
        super().__init__(num_processes=num_processes)

    def simulate_lc(self, params: Dict[str, float], cadence: ObservedCadence) -> Tuple[LightCurve, SNModel]:
        """Duplicate a plastic light-curve using the simulation model

        Args:
            params: The simulation parameters to use with ``self.model``
            cadence: The observed cadence of the returned light-curve
        """

        # Set model parameters and scale the source brightness to the desired intrinsic brightness
        model_for_sim = copy(self.sim_model)
        model_for_sim.update({p: v for p, v in params.items() if p in model_for_sim.param_names})
        model_for_sim.set_source_peakabsmag(self.abs_mb, 'standard::b', 'AB', cosmo=self.cosmo)

        # Simulate the light-curve. Make sure to include model parameters as meta data
        duplicated = model_for_sim.simulate_lc(cadence, scatter=self.add_scatter, fixed_snr=self.fixed_snr)

        # Rescale the light-curve using the reference star catalog if provided
        if self.catalog is not None:
            duplicated = self.catalog.calibrate_lc(duplicated, ra=params['ra'], dec=params['dec'])

        return duplicated, model_for_sim

    def action(self) -> None:
        """Simulate light-curves with atmospheric effects"""

        for packet in self.input.iter_get():
            try:
                light_curve, model = self.simulate_lc(packet.sim_params, packet.cadence)

            except Exception as excep:
                packet.message = f'{self.__class__.__name__}: {repr(excep)}'
                self.failure_output.put(packet)

            else:
                packet.light_curve = light_curve
                packet.sim_params['x0'] = model['x0']
                self.success_output.put(packet)


class FitLightCurves(Node):
    """Pipeline node for fitting simulated light-curves

    Connectors:
        input: A Pipeline Packet
        success_output: Emits pipeline packets with successful fit results
        failure_output: Emits pipeline packets for cases where the fitting procedure failed
    """

    def __init__(
            self, sn_model: SNModel, vparams: List[str], bounds: Dict = None, num_processes: int = 1
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when fitting light-curves
            vparams: List of parameter names to vary in the fit
            bounds: Bounds to impose on ``fit_model`` parameters when fitting light-curves
            num_processes: Number of processes to allocate to the node
        """

        self.sn_model = sn_model
        self.vparams = vparams
        self.bounds = bounds

        # Node Connectors
        self.input = Input('Simulated Light-Curve')
        self.success_output = Output('Fitting Success')
        self.failure_output = Output('Fitting Failure')
        super(FitLightCurves, self).__init__(num_processes=num_processes)

    def fit_lc(self, light_curve: LightCurve, initial_guess: Dict[str, float]) -> SNFitResult:
        """Fit the given light-curve

        Args:
            light_curve: The light-curve to fit
            initial_guess: Parameters to use as the initial guess in the chi-squared minimization

        Returns:
            - The optimization result
            - A copy of the model with parameter values set to minimize the chi-square
        """

        # Use the true light-curve parameters as the initial guess
        model = copy(self.sn_model)
        model.update({k: v for k, v in initial_guess.items() if k in self.sn_model.param_names})

        bounds = copy(self.bounds)
        if 't0' in bounds:
            model_t0 = model['t0']
            lower_t0, upper_t0 = bounds['t0']
            bounds['t0'] = (model_t0 + lower_t0, model_t0 + upper_t0)

        return model.fit_lc(
            light_curve, self.vparams, bounds=bounds,
            guess_t0=False, guess_amplitude=False, guess_z=False)

    def action(self) -> None:
        """Fit light-curves"""

        for packet in self.input.iter_get():
            try:
                packet.fit_result = self.fit_lc(packet.light_curve, packet.sim_params)
                packet.covariance = packet.fit_result.salt_covariance_linear()

            except Exception as excep:
                packet.message = f'{self.__class__.__name__}: {repr(excep)}'
                self.failure_output.put(packet)

            else:
                packet.message = f'{self.__class__.__name__}: {packet.fit_result.message}'
                self.success_output.put(packet)


class WritePipelinePacket(Target):
    """Pipeline node for writing pipeline packets to disk

    Connectors:
        input: A pipeline packet
    """

    def __init__(self, out_path: Union[str, Path], write_lc_sims: bool = False, num_processes=1) -> None:
        """Output node for writing HDF5 data to disk

        This node can only be run using a single process.

        Args:
            out_path: Path to write data to in HDF5 format
            write_lc_sims: Whether to include simulated light-curves in the data written to disk
        """

        # Make true to raise errors instead of converting them to warnings
        self.input = Input('Data To Write')
        self.write_lc_sims = write_lc_sims
        self.debug = False

        self.out_path = Path(out_path)
        self.file_store: Optional[pd.HDFStore] = None
        self._num_results_per_file = 10_000
        self._num_results_in_current_file = 0
        self._current_file_id = 0
        super().__init__(num_processes=num_processes)

    def _rotate_output_file(self) -> None:
        """Have the running process close the current output file and start writing to a new one

        Once files get too large the write performance starts to suffer.
        We address this by closing the current file, incrementing
        a number in the output file path, and writing data to that new path
        """

        if self._num_results_in_current_file < self._num_results_per_file:
            return

        if self.file_store is not None:
            self.file_store.close()

        # Update output file path
        old_id = self._current_file_id
        self._current_file_id += 1
        new_stem = self.out_path.stem.replace(f'_fn{old_id}', f'_fn{self._current_file_id}')
        self.out_path = self.out_path.with_stem(new_stem)

        # noinspection PyTypeChecker
        self.file_store = pd.HDFStore(self.out_path, mode='w')
        self._num_results_in_current_file = 0

    def _write_packet(self, packet: PipelinePacket) -> None:
        """Write a pipeline packet to the output file"""

        self._rotate_output_file()

        # We are taking the simulated parameters as guaranteed to exist
        self.file_store.append('simulation/params', packet.sim_params_to_pandas())
        self.file_store.append('message', packet.packet_status_to_pandas().astype(str), min_itemsize={'snid': 10, 'message': 250})

        if self.write_lc_sims and packet.light_curve is not None:
            self.file_store.put(f'simulation/lcs/{packet.snid}', packet.light_curve.to_pandas())

        if packet.fit_result is not None:
            self.file_store.append('fitting/params', packet.fitted_params_to_pandas())

        if packet.covariance is not None:
            self.file_store.put(f'fitting/covariance/{packet.snid}', packet.covariance)

        self._num_results_in_current_file += 1

    def setup(self) -> None:
        """Open a file accessor object"""

        # If we are writing data to disk in parallel, add the process id to
        # prevent multiple processes writing to the same file
        if self.num_processes > 1:
            import multiprocessing
            pid = hex(id(multiprocessing.current_process()))  # Use hex for shorter filename
            self.out_path = self.out_path.with_suffix(f'.{pid}.h5')

        self.out_path = self.out_path.with_stem(self.out_path.stem + f'_fn{self._current_file_id}')

        # noinspection PyTypeChecker
        self.file_store = pd.HDFStore(self.out_path, mode='w')

    def teardown(self) -> None:
        """Close any open file accessors"""

        self.file_store.close()
        self.file_store = None

    def action(self) -> None:
        """Write data from the input connector to disk"""

        for packet in self.input.iter_get():
            try:
                self._write_packet(packet)

            except Exception as excep:
                if self.debug:
                    raise

                warnings.warn(f'{self.__class__.__name__}: {repr(excep)}')
