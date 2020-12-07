#!/usr/bin/env python3

"""Commandline interface for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import argparse
import os
import sys
from pathlib import Path

from astropy.table import Table
from pwv_kpno.defaults import ctio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snat_sim import filters, models
from snat_sim.fitting_pipeline import FittingPipeline

os.environ.setdefault('CADENCE_SIMS', '/mnt/md0/sn-sims')
CTIO_PWV_MODEL = models.PWVModel.from_suominet_receiver(ctio, 2016, [2017])
SALT2_PARAMS = ('z', 't0', 'x0', 'x1', 'c')


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


def create_pwv_effect(pwv_variability):
    """Create a PWV transmission effect for use with supernova models

    If ``pwv_variability`` is numeric, return a ``StaticPWVTrans`` object
    set to the given PWV concentration (in mm). If ``pwv_variability`` equals
    ``epoch``, return a ``VariablePWVTrans`` constructed from the CTIO receiver
    (using 2016 data supplemented with 2017).

    Args:
        pwv_variability (str, Numeric): How to vary PWV as a function of time

    Returns:
        A propagation effect usable with a supernova model object
    """

    # Keep a fixed PWV concentration
    if isinstance(pwv_variability, (float, int)):
        transmission_effect = models.StaticPWVTrans()
        transmission_effect.set(pwv=pwv_variability)
        return transmission_effect

    # Model PWV continuously over the year using CTIO data
    elif pwv_variability == 'epoch':
        return models.VariablePWVTrans(CTIO_PWV_MODEL)

    else:
        raise NotImplementedError(f'Unknown variability: {pwv_variability}')


def run_pipeline(cli_args):
    """Run the fitting pipeline for a given cadence

    Args:
        cli_args (Namespace): Parse command line arguments
    """

    print('Creating simulation model...')
    sn_model_sim = models.SNModel(cli_args.source)
    sn_model_sim.add_effect(
        effect=create_pwv_effect(cli_args.sim_variability),
        name='',
        frame='obs')

    print('Creating fitting model...')
    sn_model_fit = models.SNModel(cli_args.source)
    sn_model_fit.add_effect(
        effect=create_pwv_effect(cli_args.fit_variability),
        name='',
        frame='obs')

    print('Instantiating pipeline...')
    pipeline = FittingPipeline(
        cadence=cli_args.cadence,
        sim_model=sn_model_sim,
        fit_model=sn_model_fit,
        vparams=cli_args.vparams,
        out_path=cli_args.out_path,
        quality_callback=passes_quality_cuts,
        pool_size=cli_args.pool_size,
        iter_lim=cli_args.iter_lim,
        ref_stars=cli_args.ref_stars,
        pwv_model=CTIO_PWV_MODEL
    )

    print('I/O Processes: 2')
    print(f'Simulation Processes:', pipeline.simulation_pool_size)
    print('Fitting Processes:', pipeline.fitting_pool_size)
    pipeline.run()


def create_cli_parser():
    """Return a command line argument parser"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cadence',
        type=str,
        required=True,
        help='Cadence to use when simulating light-curves'
    )

    parser.add_argument(
        '-t', '--source',
        type=str,
        default='salt2-extended',
        help='The name of the spectral template to use when simulating AND fitting'
    )

    parser.add_argument(
        '-v', '--vparams',
        type=str,
        default=('x0', 'x1', 'c'),
        nargs='+',
        help='Parameters to vary when fitting'
    )

    parser.add_argument(
        '-s', '--sim_variability',
        type=str,
        required=True,
        help='Rate at which to vary PWV in simulated light-curves'
    )

    parser.add_argument(
        '-f', '--fit_variability',
        type=str,
        required=True,
        help='Rate at which to vary assumed PWV when fitting light-curves'
    )

    parser.add_argument(
        '-p', '--pool_size',
        type=int,
        default=None,
        help='Total number of workers to spawn. Defaults to CPU count'
    )

    parser.add_argument(
        '-i', '--iter_lim',
        type=int,
        default=float('inf'),
        help='Limit number of processed light-curves (Useful for profiling)'
    )

    parser.add_argument(
        '-r', '--ref_stars',
        type=str,
        default=('G2', 'M5', 'K2'),
        nargs='+',
        help='Reference star(s) to calibrate simulated SNe against'
    )

    parser.add_argument(
        '-o', '--out_path',
        type=Path,
        required=True,
        help='Output file path (in CSV format)'
    )

    parser.set_defaults(func=run_pipeline)
    return parser


if __name__ == '__main__':
    filters.register_lsst_filters()
    parsed_args = create_cli_parser().parse_args()

    # Types cast PWV variability into float
    if parsed_args.fit_variability.isnumeric():
        parsed_args.fit_variability = float(parsed_args.fit_variability)

    if parsed_args.sim_variability.isnumeric():
        parsed_args.sim_variability = float(parsed_args.sim_variability)

    print(parsed_args)
    sys.exit(0)
    parsed_args.func(parsed_args)
