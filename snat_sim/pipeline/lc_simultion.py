from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import *
from typing import Dict

import h5py
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table
from egon.connectors import Input, Output
from egon.nodes import Node, Source, Target

from .data_model import PipelineResult
from .. import constants as const
from ..models import AbstractVariablePWVEffect, ObservedCadence, PWVModel, SNModel, StaticPWVTrans
from ..reference_stars import VariableCatalog

__all__ = ['SimulateLightCurves', 'SimulationToDisk', 'SimulationFromDisk']


class SimulateLightCurves(Node):
    """Pipeline node for simulating light-curves based on PLaSTICC cadences

    Connectors:
        plasticc_data_input: PLaSTICC light-curves as ``astropy.Table`` objects
        simulation_output: Simulated light-curves as  ``astropy.Table`` objects
    """

    def __init__(
            self,
            sn_model: SNModel,
            catalog: VariableCatalog = None,
            num_processes: int = 1,
            add_scatter=True,
            abs_mb: float = const.betoule_abs_mb,
            cosmo=const.betoule_cosmo,
            include_pwv: bool = False
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
        self.abs_mb = abs_mb
        self.cosmo = cosmo
        self.include_pwv_col = include_pwv

        # Node connectors
        self.plasticc_data_input = Input()
        self.simulation_output = Output()
        self.failure_result_output = Output()
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
        duplicated = model_for_sim.simulate_lc(cadence, scatter=self.add_scatter)
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

        for params, cadence in self.plasticc_data_input.iter_get():
            try:
                duplicated_lc = self.duplicate_plasticc_lc(params, cadence)

            except Exception as e:
                result = PipelineResult(
                    params['SNID'], sim_params=params,
                    message=f'{self.__class__.__name__}: {e}')
                self.failure_result_output.put(result)

            else:
                self.simulation_output.put(duplicated_lc)


class SimulationToDisk(Target):
    """Pipeline node for writing simulated light-curves to disk

    Connectors:
        simulation_input: Simulated light-curves
    """

    def __init__(self, out_path: Union[str, Path], num_processes: int = 1) -> None:
        """Write simulated light-curves to disk

        Args:
            out_path: HDF5 path (``.h5``) to write results to
        """

        self.out_path = Path(out_path).with_suffix('.h5')
        self.simulation_input = Input()
        super(SimulationToDisk, self).__init__(num_processes)

    def setup(self) -> None:
        """Ensure the output directory exists"""

        self.out_path.parent.mkdir(exist_ok=True, parents=False)

    def action(self) -> None:
        """Write simulated light-curves to disk"""

        path_str = str(self.out_path)
        for lc in self.simulation_input.iter_get():
            write_table_hdf5(table=lc, output=path_str, path=lc.meta['SNID'], append=True)


class SimulationFromDisk(Source):
    """Pipeline node for writing simulated light-curves to disk

    Connectors:
        simulation_input: Simulated light-curves
    """

    def __init__(self, int_path: Union[str, Path], num_processes: int = 1) -> None:
        """Write simulated light-curves to disk

        Args:
            int_path: HDF5 path (``.h5``) to read results from
        """

        if num_processes > 1:
            raise ValueError('Number of forked processes for loading cached sims cannot exceed 1')

        self.sim_data: h5py.File
        self.out_path = Path(int_path).with_suffix('.h5')
        self.simulation_output = Output()

        super().__init__(num_processes)

    def setup(self) -> None:
        """Establish I/O for input file"""

        self.sim_data = h5py.File(self.out_path)

    def action(self) -> None:
        """Write simulated light-curves to disk"""

        for key in self.sim_data.keys():
            data = Table(np.array(self.sim_data[key]))
            cols_to_cast = [c for c, cdata in data.columns.items() if cdata.dtype.type == np.bytes_]
            for column in cols_to_cast:
                data[column] = data[column].astype(str)

            self.simulation_output.put(data)
