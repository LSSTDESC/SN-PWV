#!/usr/bin/env python3

"""Commandline interface for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import argparse
import sys
from pathlib import Path

from pwv_kpno.defaults import ctio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snat_sim import filters, models
from snat_sim.fitting_pipeline import FittingPipeline
from tests.mock import create_constant_pwv_model


def create_pwv_effect(pwv_variability):
    """Create a ``VariablePWVTrans`` object

    Args:
        pwv_variability (str, numeric): How to vary PWV as a function of time

    Returns:
        An instantiated ``VariablePWVTrans`` object
    """

    # Keep a fixed PWV concentration
    if isinstance(pwv_variability, (float, int)):
        pwv_model = create_constant_pwv_model(pwv_variability)

    # Model PWV continuously over the year using CTIO data
    elif pwv_variability == 'epoch':
        pwv_model = models.PWVModel.from_suominet_receiver(ctio, 2016, [2017])

    else:
        raise NotImplementedError(f'Unknown variability: {pwv_variability}')

    variable_pwv_effect = models.VariablePWVTrans(pwv_model)
    variable_pwv_effect.set(res=5)
    return variable_pwv_effect


def create_sn_model(source='salt2-extended', pwv_variability=None):
    """Create a supernova model with optional PWV effects

    Args:
        source           (str, Source): Spectral template to use for the SN model
        pwv_variability (str, numeric): How to vary PWV as a function of time

    Returns:
        An instantiated ``snat_sim`` supernova model
    """

    model = models.Model(source=source)
    if pwv_variability:
        model.add_effect(
            effect=create_pwv_effect(pwv_variability),
            name='',
            frame='obs')

    return model


def run_pipeline(source, cadence, sim_variability, fit_variability, **kwargs):
    """Run the fitting pipeline for a given cadence"""

    sim_model = create_sn_model(source, sim_variability)
    fit_model = create_sn_model(source, fit_variability)

    FittingPipeline(
        cadence,
        sim_model,
        fit_model,
        vparams,
        gain=20,
        skynr=100,
        quality_callback=None,
        max_queue=25,
        pool_size=None,
        iter_lim=float('inf'),
        ref_stars=None,
        pwv_model=None).run(out_path=out_path)


def create_cli_parser():
    """Return a command line argument parser"""

    # Todo: Add the following arguments
    #    X    cadence
    #         sim_variability
    #         fit_variability
    #    X    source = slat2-extended
    #         vparams
    #         max_queue = 25
    #         pool_size = None
    #         iter_lim = float('inf')
    #         ref_stars = ('G2', 'M5', 'K2')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cadence',
        type=str,
        required=True,
        help='Cadence to use when simulating light-curves.'
    )

    parser.add_argument(
        '-s', '--source',
        type=str,
        default='salt2-extended',
        help='The name of the spectral template to use when simulating AND fitting.'
    )

    parser.set_defaults(func=run_pipeline)
    return parser


if __name__ == '__main__':
    filters.register_lsst_filters()
    cli_args = create_cli_parser().parse_args()
    cli_args.func(cli_args)
