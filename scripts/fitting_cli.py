#!/usr/bin/env python3

"""Commandline interface for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import argparse
import sys
from pathlib import Path

from astropy.table import Table
from pwv_kpno.gps_pwv import GPSReceiver

sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent.parent))
from snat_sim import models
from snat_sim.pipeline import FittingPipeline

SALT2_PARAMS = ('z', 't0', 'x0', 'x1', 'c')
SUOMINET_VALUES = ('press', 'temp', 'rh', 'zenith_delay')


# Todo: This should be in the snat_sim package so it can be tested
def passes_quality_cuts(light_curve):
    """Return whether light-curve has 2+ two bands each with 1+ data point with SNR > 5

    Args:
        light_curve (Table): Astropy table with sncosmo formatted light-curve data

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


class AdvancedNamespace(argparse.Namespace):
    """Represents parsed command line arguments cast into friendly object types"""

    def _create_pwv_effect(self, pwv_variability):
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
            transmission_effect.set(pwv=float(parsed_args.fit_variability))
            return transmission_effect

        # Model PWV continuously over the year using CTIO data
        elif pwv_variability == 'epoch':
            return models.VariablePWVTrans(self.pwv_model)

        elif pwv_variability == 'seasonal':
            return models.SeasonalPWVTrans(self.pwv_model)

        raise NotImplementedError(f'Unknown variability: {pwv_variability}')

    @property
    def pwv_model(self):
        """Build a PWV model based on the command line argument"""

        print('Building PWV Model...')
        data_cuts = dict()
        for value in SUOMINET_VALUES:
            if param_bound := getattr(self, f'cut_{value}', None):
                data_cuts[value] = param_bound

        primary_year, *supp_years = self.pwv_model_years
        receiver = GPSReceiver(self.receiver_id, data_cuts=data_cuts)
        return models.PWVModel.from_suominet_receiver(receiver, primary_year, supp_years)

    @property
    def fitting_bounds(self):
        """Parameter boundaries to enforce when fitting light-curves

        Returns:
            A dictionary {<Param Name>: [<Lower Bound>, <Upper Bound>]}
        """

        fitting_bounds = dict()
        for param in SALT2_PARAMS:
            if param_bound := getattr(self, f'bound_{param}', None):
                fitting_bounds[param] = param_bound

        return fitting_bounds

    @property
    def simulation_model(self):
        """Return the Supernova model used for fitting light-curves

        Returns:
            An SNModel object with atmospheric propagation effects
        """

        propagation_effect = self._create_pwv_effect(self.sim_variability)

        print('Building supernova simulation model...')
        return models.SNModel(
            source=self.sim_source,
            effects=[propagation_effect],
            effect_names=[''],
            effect_frames=['obs'])

    @property
    def fitting_model(self):
        """Return the Supernova model used for simulating light-curves

        Returns:
            An SNModel object with atmospheric propagation effects
        """

        propagation_effect = self._create_pwv_effect(self.fit_variability)

        print('Building supernova fitting model...')
        return models.SNModel(
            source=self.sim_source,
            effects=[propagation_effect],
            effect_names=[''],
            effect_frames=['obs'])


def run_pipeline(command_line_args):
    """Run the analysis pipeline

    Args:
        command_line_args (AdvancedNamespace): Parsed command line arguments
    """

    print('Instantiating pipeline...')
    pipeline = FittingPipeline(
        cadence=command_line_args.cadence,
        sim_model=command_line_args.simulation_model,
        fit_model=command_line_args.fitting_model,
        vparams=command_line_args.vparams,
        out_path=command_line_args.out_path,
        simulation_pool=command_line_args.sim_pool_size,
        fitting_pool=command_line_args.fit_pool_size,
        bounds=command_line_args.fitting_bounds,
        quality_callback=passes_quality_cuts,
        iter_lim=command_line_args.iter_lim,
        ref_stars=command_line_args.ref_stars,
        pwv_model=command_line_args.pwv_model
    )

    pipeline.run()


def create_cli_parser():
    """Return a command line argument parser"""

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=run_pipeline)

    parser.add_argument(
        '-c', '--cadence',
        type=str,
        required=True,
        help='Observational cadence to assume for the LSST.'
    )

    parser.add_argument(
        '-s', '--sim_pool_size',
        type=int,
        default=1,
        help='Total number of workers to spawn for simulating supernova light-curves.'
    )

    parser.add_argument(
        '-f', '--fit_pool_size',
        type=int,
        default=1,
        help='Total number of workers to spawn for fitting light-curves.'
    )

    parser.add_argument(
        '-i', '--iter_lim',
        type=int,
        default=float('inf'),
        help='Exit pipeline after processing the given number of light-curves (Useful for profiling).'
    )

    parser.add_argument(
        '-o', '--out_path',
        type=Path,
        required=True,
        help='Output file path (a .csv extension is enforced).'
    )

    #######################################################################
    # Light-curve simulation
    #######################################################################

    simulation_group = parser.add_argument_group(
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

    fitting_group = parser.add_argument_group(
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

    pwv_modeling_group = parser.add_argument_group(
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
        '--cut_pwv',
        type=float,
        nargs=2,
        default=[0, 30],
        help='Only use measured data points with a PWV value within the given bounds (units of millimeters)'
    )

    data_cut_names = ('surface pressure', 'temperature', 'relative humidity', 'zenith delay')
    data_cut_units = ('Millibars', 'Centigrade', 'Percentage', 'Millimeters')
    for arg, name, unit in zip(SUOMINET_VALUES, data_cut_names, data_cut_units):
        pwv_modeling_group.add_argument(
            f'--cut_{arg}',
            type=float,
            nargs=2,
            help=f'Only use measured data points with a {name} value within the given bounds (units of {unit})'
        )

    return parser


if __name__ == '__main__':
    parsed_args = create_cli_parser().parse_args(namespace=AdvancedNamespace())
    parsed_args.func(parsed_args)
