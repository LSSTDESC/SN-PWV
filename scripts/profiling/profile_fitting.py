import os
from pathlib import Path

from pwv_kpno.defaults import ctio

from snat_sim import models
from snat_sim.pipeline import FittingPipeline

os.environ['CADENCE_SIMS'] = str(Path(__file__).resolve().parent / 'data')


def setup_pipeline():
    pwv_model = models.PWVModel.from_suominet_receiver(ctio, 2016, [2017])
    propagation_effect = models.VariablePWVTrans(pwv_model)

    sn_model_sim = models.SNModel(
        source='salt2-extended',
        effects=[propagation_effect],
        effect_names=[''],
        effect_frames=['obs'])

    sn_model_fit = models.SNModel(
        source='salt2-extended',
        effects=[propagation_effect],
        effect_names=[''],
        effect_frames=['obs'])

    return FittingPipeline(
        cadence='alt_sched',
        sim_model=sn_model_sim,
        fit_model=sn_model_fit,
        vparams=['x0', 'x1', 'c'],
        simulation_pool=4,
        fitting_pool=6,
        out_path='./test.csv',
        max_queue=1000,
        iter_lim=100
    )


if __name__ == '__main__':
    pipeline = setup_pipeline()

    print('Loading data')
    pipeline.load_plastic.execute()

    print('Simulating data')
    pipeline.simulate_light_curves.execute()

    print('Fitting data')
    pipeline.fit_light_curves.execute()
