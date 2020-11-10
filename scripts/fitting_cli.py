#!/usr/bin/env python3

"""Commandline interface for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import argparse
import sys
from pathlib import Path

from astropy.table import Table
from pwv_kpno.defaults import ctio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snat_sim import filters, models
from snat_sim.fitting_pipeline import FittingPipeline
from tests.mock import create_constant_pwv_model


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


def create_pwv_model(pwv_variability):
    """Create a ``PWVModel`` object

    Args:
        pwv_variability (str, numeric): How to vary PWV as a function of time

    Returns:
        An instantiated ``PWVModel`` object
    """

    # Keep a fixed PWV concentration
    if isinstance(pwv_variability, (float, int)):
        return create_constant_pwv_model(pwv_variability)

    # Model PWV continuously over the year using CTIO data
    elif pwv_variability == 'epoch':
        return models.PWVModel.from_suominet_receiver(ctio, 2016, [2017])

    else:
        raise NotImplementedError(f'Unknown variability: {pwv_variability}')


def create_sn_model(source='salt2-extended', pwv_model=None):
    """Create a supernova model with optional PWV effects

    Args:
        source (str, Source): Spectral template to use for the SN model
        pwv_model (PWVModel): How to vary PWV as a function of time

    Returns:
        An instantiated ``snat_sim`` supernova model
    """

    model = models.Model(source=source)
    if pwv_model is not None:
        model.add_effect(
            effect=models.VariablePWVTrans(pwv_model),
            name='',
            frame='obs')

    return model


def run_pipeline(cli_args):
    """Run the fitting pipeline for a given cadence

    Args:
        cli_args (Namespace): Parse command line arguments
    """

    pwv_model_sim = create_pwv_model(cli_args.sim_variability)
    sn_model_sim = create_sn_model(cli_args.source, pwv_model_sim)

    pwv_model_fit = create_pwv_model(cli_args.fit_variability)
    sn_model_fit = create_sn_model(cli_args.source, pwv_model_fit)

    pipeline = FittingPipeline(
        cli_args.cadence,
        sn_model_sim,
        sn_model_fit,
        cli_args.vparams,
        quality_callback=passes_quality_cuts,
        pool_size=cli_args.pool_size,
        iter_lim=cli_args.iter_lim,
        ref_stars=cli_args.ref_stars,
        pwv_model=pwv_model_sim
    )

    pipeline.run(out_path=cli_args.out_path)


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
        type=str,
        required=True,
        help='Output file path (in CSV format)'
    )

    parser.set_defaults(func=run_pipeline)
    return parser


if __name__ == '__main__':
    filters.register_lsst_filters()
    cli_args = create_cli_parser().parse_args()
    cli_args.func(cli_args)
