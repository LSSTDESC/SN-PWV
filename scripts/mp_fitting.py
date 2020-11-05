#!/usr/bin/env python3

"""Multiprocess script for simulating light-curves with atmospheric effects
and then fitting them with a given SN model.
"""

import sys
from pathlib import Path

import sncosmo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snat_sim import modeling, plasticc, filters
from snat_sim.fitting_pipeline import FittingPipeline

filters.register_lsst_filters()

if __name__ == '__main__':

    available_cadences = plasticc.get_available_cadences()
    cadence_name = sys.argv[1]
    if cadence_name not in available_cadences:
        raise ValueError(f'Cadence {cadence_name} not available from local cadences: {available_cadences}')

    # Characterize the atmospheric variability
    # Set PWV to a constant while developing
    pwv_interpolator = lambda *args: 5
    variable_pwv_effect = modeling.VariablePWVTrans(pwv_interpolator)
    variable_pwv_effect.set(res=5)

    # Build models with and without atmospheric effects
    model_without_pwv = sncosmo.Model('Salt2-extended')
    model_with_pwv = modeling.Model(
        source='salt2-extended',
        effects=[variable_pwv_effect],
        effect_names=[''],
        effect_frames=['obs']
    )

    output = Path(__file__).resolve().parent.parent / 'results' / 'fit_results.csv'
    FittingPipeline(
        cadence=cadence_name,
        sim_model=model_with_pwv,
        fit_model=model_without_pwv,
        vparams=['x0', 'x1', 'c'],
        pool_size=10,
        iter_lim=100
    ).run(out_path=output)
