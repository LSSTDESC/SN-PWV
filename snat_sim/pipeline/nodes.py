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

from astropy.cosmology.core import Cosmology
from astropy.table import Table
from egon.connectors import Input, Output
from egon.nodes import Node, Source, Target

from .. import constants as const
from ..models import AbstractVariablePWVEffect, ObservedCadence, PWVModel, SNModel, StaticPWVTrans
from ..pipeline.data_model import PipelinePacket
from ..plasticc import PLaSTICC
from ..reference_stars import VariableCatalog

warnings.simplefilter('ignore', category=DeprecationWarning)


class LoadPlasticcCadence(Source):
    """Pipeline node for loading PLaSTICC cadence data from disk

    Connectors:
        output: Emits cadence data for individual SN simulations as ``ObservedCadence`` objects
    """

    def __init__(self, cadence: str, model: int = 11, iter_lim: int = float('inf')) -> None:
        """Source node for loading PLaSTICC cadence data from disk

        This node can only be run using a single process.

        Args:
            cadence: Cadence name to load from disk
            model: The PLaSTICC supernova model to load simulation for (Default is model 11 - Normal SNe)
            iter_lim: Exit after loading the given number of light-curves
        """

        self.cadence = PLaSTICC(cadence, model)
        self.iter_lim = iter_lim

        # Node connectors
        self.output = Output('Loading Cadence Output')
        super().__init__(num_processes=1)

    def action(self) -> None:
        """Load PLaSTICC cadence data from disk"""

        for light_curve in self.cadence.iter_lc(iter_lim=self.iter_lim):
            params, cadence = ObservedCadence.from_plasticc(light_curve)
            packet = PipelinePacket(light_curve.meta['SNID'], params, cadence)
            self.output.put(packet)


class SimulateLightCurves(Node):
    """Pipeline node for simulating light-curves based on PLaSTICC cadences

    Connectors:
        cadence_data_input: PLaSTICC light-curves as ``astropy.Table`` objects
        simulation_output: Simulated light-curves as  ``astropy.Table`` objects
    """

    def __init__(
            self,
            sn_model: SNModel,
            catalog: VariableCatalog = None,
            num_processes: int = 1,
            add_scatter: bool = True,
            fixed_snr: Optional[float] = None,
            abs_mb: float = const.betoule_abs_mb,
            cosmo: Cosmology = const.betoule_cosmo,
            include_pwv: bool = True
    ) -> None:
        """Fit light-curves using multiple processes and combine results into an output file

        Args:
            sn_model: Model to use when simulating light-curves
            catalog: Optional reference start catalog to calibrate simulated flux values to
            num_processes: Number of processes to allocate to the node
            abs_mb: The absolute B-band magnitude of the simulated SNe
            cosmo: Cosmology to assume in the simulation
        """

        self.sim_model = sn_model
        self.catalog = catalog
        self.add_scatter = add_scatter
        self.fixed_snr = fixed_snr
        self.abs_mb = abs_mb
        self.cosmo = cosmo
        self.include_pwv_col = include_pwv

        # Node connectors
        self.cadence_data_input = Input('Simulation Cadence Input')
        self.success_output = Output('Simulation Success')
        self.failure_output = Output('Simulation Failure')
        super().__init__(num_processes)

    def duplicate_plasticc_lc(self, params: Dict[str, float], cadence: ObservedCadence) -> Table:
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
        duplicated.meta = params
        duplicated.meta.update(dict(zip(model_for_sim.param_names, model_for_sim.parameters)))

        # Rescale the light-curve using the reference star catalog if provided
        if self.catalog is not None:
            duplicated = self.catalog.calibrate_lc(
                duplicated, duplicated['time'], ra=params['ra'], dec=params['dec'])

        # Add the simulated PWV concentration if there is a variable PWV transmission effect.
        if self.include_pwv_col:
            self.add_pwv_columns_to_table(duplicated, model_for_sim, ra=params['ra'], dec=params['dec'])

        return duplicated

    @staticmethod
    def add_pwv_columns_to_table(light_curve: Table, model_for_sim: SNModel, ra: float, dec: float) -> None:
        """Add columns for PWV and Airmass to a light-curve table

        Args:
            light_curve: The simulated light-curve table
            model_for_sim: The model used to simulate the light-curve
            ra: The Right Ascension of the supernova
            dec: The declination  of the supernova
        """

        for effect in model_for_sim.effects:
            if isinstance(effect, AbstractVariablePWVEffect):
                light_curve['pwv'] = effect.assumed_pwv(light_curve['time'])
                light_curve['airmass'] = PWVModel.calc_airmass(light_curve['time'], ra=ra, dec=dec)
                break

            if isinstance(effect, StaticPWVTrans):
                light_curve['pwv'] = effect['pwv']
                light_curve['airmass'] = 1
                break

    def action(self) -> None:
        """Simulate light-curves with atmospheric effects"""

        for packet in self.cadence_data_input.iter_get():
            try:
                packet.light_curve = self.duplicate_plasticc_lc(
                    packet.sim_params, packet.cadence
                ).to_pandas()

            except Exception as e:
                packet.message = f'{self.__class__.__name__}: {e}'
                self.failure_output.put(packet)

            else:
                self.success_output.put(packet)


class FitLightCurves(Node):
    """Pipeline node for fitting simulated light-curves

    Connectors:
        light_curves_input: Light-curves to fit
        fit_results_output: Fit results as a list
    """

    def __init__(
            self,
            sn_model: SNModel,
            vparams: List[str],
            bounds: Dict = None,
            num_processes: int = 1
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
        self.light_curves_input = Input('Fitting Light-Curve Input')
        self.success_output = Output('Fitting Success')
        self.failure_output = Output('Fitting Failure')
        super(FitLightCurves, self).__init__(num_processes)

    def fit_lc(self, packet: PipelinePacket):
        """Fit the given light-curve

        Args:
            packet: A pipeline data packet

        Returns:
            - The optimization result represented as a ``Result`` object
            - A copy of ``self.model`` with parameter values set to optimize the chi-square
        """

        # Use the true light-curve parameters as the initial guess
        model = copy(self.sn_model)
        model.update({k: v for k, v in packet.sim_params.items() if k in self.sn_model.param_names})

        return model.fit_lc(
            packet.light_curve, self.vparams, bounds=self.bounds,
            guess_t0=False, guess_amplitude=False, guess_z=False)

    def action(self) -> None:
        """Fit light-curves"""

        for packet in self.light_curves_input.iter_get():
            try:
                packet.fit_result, packet.fitted_model = self.fit_lc(packet)

            except Exception as excep:
                packet.message = f'{self.__class__.__name__}: {excep}'
                self.failure_output.put(packet)

            else:
                packet.message = packet.fit_result.message
                self.success_output.put(packet)


class WritePipelinePacket(Target):
    """Pipeline node for writing pandas objects to disk in HDF5 format

    Connectors:
        fit_results_input: Expects a tuple with the HDF5 key and the data to write to that key
    """

    def __init__(self, out_path: Union[str, Path]) -> None:
        """Output node for writing HDF5 data to disk

        This node can only be run using a single process.

        Args:
            out_path: Path to write data to in HDF5 format
        """

        self.out_path = Path(out_path)
        self.data_input = Input('Writing To Disk Input')
        super().__init__(num_processes=1)

    def action(self) -> None:
        """Write data from the input connector to disk"""

        for packet in self.data_input.iter_get():
            # We are taking the simulated parameters as guaranteed to exist
            packet.sim_params_to_pandas().to_hdf(self.out_path, f'simulation/params/{packet.snid}')

            fit_data = packet.fit_result_to_pandas()  # This will be a masked dataframe if fit results are not available
            fit_data.to_hdf(self.out_path, f'simulation/params', format='Table', append=True)

            if packet.light_curve is not None:
                packet.light_curve.to_hdf(self.out_path, f'simulation/lcs/{packet.snid}')

                # If the simulation failed then there is nothing to write
                # so we maintain the same indented scope for the next calculation.
                # The covariance need to be calculated before being written
                if packet.fit_result is not None:
                    covariance = packet.fit_result.salt_covariance_linear()
                    covariance.to_hdf(self.out_path, f'fitting/covariance/{packet.snid}')
