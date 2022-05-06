"""Commandline interface for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

from pwv_kpno.gps_pwv import GPSReceiver

sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent.parent))
from snat_sim import models
from snat_sim.pipeline import FittingPipeline

SALT2_PARAMS = ('z', 't0', 'x0', 'x1', 'c')
SUOMINET_VALUES = ('PWV', 'SrfcPress', 'SrfcTemp', 'SrfcRH', 'ZenithDelay')
AtmModels = Union[models.StaticPWVTrans, models.VariablePWVTrans, models.SeasonalPWVTrans]


class AdvancedNamespace(argparse.Namespace):
    """Represents parsed command line arguments cast into friendly object types"""

    def _create_pwv_effect(self, pwv_variability) -> AtmModels:
        """Create a PWV transmission effect for use with supernova models

        Note: ``pwv_variability`` should be a string!

        If ``pwv_variability`` represents a numerical value, return a
        ``StaticPWVTrans`` object set to the given PWV concentration (in mm).

        If ``pwv_variability`` equals ``epoch``, return a ``VariablePWVTrans``
        object.

        If `pwv_variability`` equals ``seasonal``, return a
        ``SeasonalPWVTrans`` object.

        Args:
            pwv_variability (str): Command line value for how to vary PWV as a function of time

        Returns:
            A propagation effect usable with a supernova model object
        """

        # The command line parser defaults to the pwv variability being a string
        # even if it is numeric. A typecast is sometimes in order.
        if pwv_variability.isnumeric():
            transmission_effect = models.StaticPWVTrans()
            transmission_effect.set(pwv=float(pwv_variability))
            return transmission_effect

        # Model PWV continuously over the year using CTIO data
        elif pwv_variability == 'epoch':
            return models.VariablePWVTrans(self.pwv_model)

        elif pwv_variability == 'seasonal':
            return models.SeasonalPWVTrans.from_pwv_model(self.pwv_model)

        raise NotImplementedError(f'Unknown variability: {pwv_variability}')

    @property
    def pwv_model(self) -> models.PWVModel:
        """Build a PWV model based on the command line argument"""

        print('Building PWV Model...')
        data_cuts = dict()
        for value in SUOMINET_VALUES:
            if param_bound := getattr(self, f'cut_{value}', None):
                data_cuts[value] = [param_bound, ]

        primary_year, *supp_years = self.pwv_model_years
        receiver = GPSReceiver(self.receiver_id, data_cuts=data_cuts)
        return models.PWVModel.from_suominet_receiver(receiver, primary_year, supp_years)

    @property
    def fitting_bounds(self) -> Dict[str, Tuple]:
        """Parameter boundaries to enforce when fitting light-curves"""

        fitting_bounds = dict()
        for param in SALT2_PARAMS:
            if param_bound := getattr(self, f'bound_{param}', None):
                fitting_bounds[param] = param_bound

        return fitting_bounds

    @property
    def simulation_model(self) -> models.SNModel:
        """Return the Supernova model used for fitting light-curves"""

        propagation_effect = self._create_pwv_effect(self.sim_variability)

        print('Building supernova simulation model...')
        return models.SNModel(
            source=self.sim_source,
            effects=[propagation_effect],
            effect_names=[''],
            effect_frames=['obs'])

    @property
    def fitting_model(self) -> models.SNModel:
        """Return the Supernova model used for simulating light-curves"""

        propagation_effect = self._create_pwv_effect(self.fit_variability)

        print('Building supernova fitting model...')
        return models.SNModel(
            source=self.fit_source,
            effects=[propagation_effect],
            effect_names=[''],
            effect_frames=['obs'])

    @property
    def catalog(self) -> models.VariableCatalog:
        """The reference star catalog to calibrate simulations with."""

        return models.VariableCatalog(self.pwv_model, *self.ref_stars)

    @property
    def add_scatter(self) -> bool:
        """Whether to include added scatter in the light-curve simulations."""

        return not self.no_scatter


class Parser(argparse.ArgumentParser):
    """Return a command line argument parser"""

    def __init__(self) -> None:
        super().__init__()
        self.add_argument(
            '-c', '--cadence',
            type=str,
            required=True,
            help='Observational cadence to assume for the LSST.'
        )

        self.add_argument(
            '-s', '--sim_pool_size',
            type=int,
            default=1,
            help='Total number of workers to spawn for simulating supernova light-curves.'
        )

        self.add_argument(
            '-f', '--fit_pool_size',
            type=int,
            default=1,
            help='Total number of workers to spawn for fitting light-curves.'
        )

        self.add_argument(
            '-w', '--write_pool_size',
            type=int,
            default=1,
            help='Total number of workers to spawn for writing data to disk.'
        )

        self.add_argument(
            '-i', '--iter_lim',
            type=int,
            default=float('inf'),
            help='Exit pipeline after processing the given number of light-curves (Useful for profiling).'
        )

        self.add_argument(
            '-o', '--out_path',
            type=Path,
            required=True,
            help='Output file path (a .csv extension is enforced).'
        )

        self.add_argument(
            '--overwrite',
            action='store_true',
            help='Flag to allow existing results to be overwritten.'
        )

        self.add_argument(
            '--write_lc_sims',
            action='store_true',
            help='Flag to include simulated light curves in the pipeline output file.'
                 ' Note the added IO may noticeably increase the pipeline runtime.'
        )

        #######################################################################
        # Light-curve simulation
        #######################################################################

        simulation_group = self.add_argument_group(
            'Light-Curve Simulation',
            description='Options for simulating supernova light-curves.')

        simulation_group.add_argument(
            '--sim_source',
            type=str,
            default='salt2-extended',
            help='The name of the sncosmo spectral template to use when simulating supernova light-curves.'
        )

        simulation_group.add_argument(
            '--sim_variability',
            type=str,
            required=True,
            help='Rate at which to vary PWV when simulating light-curves.'
                 ' Specify a numerical value for a fixed PWV concentration.'
                 ' Specify "epoch" to vary the PWV per observation.'
        )

        simulation_group.add_argument(
            '--ref_stars',
            type=str,
            default=('G2', 'M5', 'K2'),
            nargs='+',
            help='Reference star(s) to calibrate simulated SNe against.'
        )

        #######################################################################
        # Light-curve fitting
        #######################################################################

        fitting_group = self.add_argument_group(
            'Light-Curve Fitting',
            description='Options for configuring supernova light-curve fits.')

        fitting_group.add_argument(
            '--fit_source',
            type=str,
            default='salt2-extended',
            help='The name of the sncosmo spectral template to use when fitting supernova light-curves.'
        )

        fitting_group.add_argument(
            '--fit_variability',
            type=str,
            required=True,
            help='Rate at which to vary the assumed PWV when fitting light-curves.'
                 ' Specify a numerical value for a fixed PWV concentration.'
                 ' Specify "epoch" to vary the PWV per observation.'
        )

        fitting_group.add_argument(
            '--vparams',
            type=str,
            default=('x0', 'x1', 'c'),
            nargs='+',
            help='Parameters to vary when fitting light-curves.'
        )

        for param in SALT2_PARAMS:
            fitting_group.add_argument(
                f'--bound_{param}',
                type=float,
                default=None,
                nargs=2,
                help=f'Upper and lower bounds for {param} parameter when fitting light-curves (Optional).'
            )

        #######################################################################
        # PWV Modeling
        #######################################################################

        pwv_modeling_group = self.add_argument_group(
            'PWV Modeling',
            description='Options used when building the PWV variability model'
                        ' from SuomiNet GPS data.'
        )

        pwv_modeling_group.add_argument(
            '--receiver_id',
            type=str,
            default='ctio'
        )

        pwv_modeling_group.add_argument(
            '--pwv_model_years',
            type=float,
            nargs='+',
            default=[2016, 2017]
        )

        pwv_modeling_group.add_argument(
            '--cut_PWV',
            type=float,
            nargs=2,
            default=[0, 30],
            help='Only use measured data points with a PWV value within the given bounds (units of millimeters)'
        )

        data_cut_names = ('surface pressure', 'temperature', 'relative humidity', 'zenith delay')
        data_cut_units = ('Millibars', 'Centigrade', 'Percentage', 'Millimeters')
        for arg, name, unit in zip(SUOMINET_VALUES[1:], data_cut_names, data_cut_units):
            pwv_modeling_group.add_argument(
                f'--cut_{arg}',
                type=float,
                nargs=2,
                help=f'Only use measured data points with a {name} value within the given bounds (units of {unit})'
            )

        debugging_group = self.add_argument_group(
            'Debugging / Validation',
            description='Options used when debugging pipeline behavior.'
        )

        debugging_group.add_argument(
            '--no-scatter',
            action='store_true',
            help='Flag used to turn off added scatter when simulating light-curves.'
        )

        debugging_group.add_argument(
            '--fixed-snr',
            type=float,
            default=None,
            help='Simulate light-curves with a fixed signal to noise ratio.'
        )

        #######################################################################
        # Pipeline Visualization
        #######################################################################

        visualizer = self.add_argument_group(
            title='Pipeline Visualization',
            description='Optionally launch a server instance for monitoring the pipeline in real time.')

        visualizer.add_argument(
            '--visualize',
            action='store_true',
            help='Flag used to launch a web server for visualizing the current pipeline status.'
        )

        visualizer.add_argument(
            '--host',
            type=str,
            default=None,
            help='Optionally define the host IP used to serve the application from.'
        )

        visualizer.add_argument(
            '--port',
            type=int,
            default=None,
            help='Optionally define the port used to serve the application.'
        )

        visualizer.add_argument(
            '--proxy',
            type=str,
            default=None,
            help='Optionally use a proxy to serve the application to a different URL "{input}::{output}".'
        )

    def execute(self):
        """Run the analysis pipeline"""

        parsed_args = self.parse_args(namespace=AdvancedNamespace())

        print(f'Instantiating pipeline (target: {parsed_args.out_path})')
        pipeline = FittingPipeline(
            cadence=parsed_args.cadence,
            sim_model=parsed_args.simulation_model,
            fit_model=parsed_args.fitting_model,
            vparams=parsed_args.vparams,
            out_path=parsed_args.out_path,
            simulation_pool=parsed_args.sim_pool_size,
            fitting_pool=parsed_args.fit_pool_size,
            writing_pool=parsed_args.write_pool_size,
            bounds=parsed_args.fitting_bounds,
            iter_lim=parsed_args.iter_lim,
            catalog=parsed_args.catalog,
            add_scatter=parsed_args.add_scatter,
            fixed_snr=parsed_args.fixed_snr,
            overwrite=parsed_args.overwrite,
            write_lc_sims=parsed_args.write_lc_sims
        )

        pipeline.validate()
        if parsed_args.visualize:
            pipeline.run_async()
            pipeline.visualize()

        else:
            pipeline.run()
