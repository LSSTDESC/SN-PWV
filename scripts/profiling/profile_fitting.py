import os
from pathlib import Path

from scripts.fitting_cli import create_pwv_model, create_sn_model
from snat_sim.filters import register_lsst_filters
from snat_sim.fitting_pipeline import FittingPipeline

register_lsst_filters(force=True)
os.environ['CADENCE_SIMS'] = str(Path(__file__).resolve().parent / 'data')

if __name__ == '__main__':
    sn_model_sim = create_sn_model('salt2-extended', create_pwv_model(4, cache=False), cache=False)
    sn_model_fit = create_sn_model('salt2-extended', create_pwv_model(4, cache=True), cache=True)
    pipeline = FittingPipeline(
        cadence='alt_sched',
        sim_model=sn_model_sim,
        fit_model=sn_model_fit,
        vparams=['x0', 'x1', 'c'],
        pool_size=7,
        out_path='./test.csv',
        max_queue=1000,
        iter_lim=10
    )

    print('Loading data')
    pipeline._load_queue_plasticc_lc()
    print('Simulating data')
    pipeline._duplicate_light_curves()
    print('Fitting data')
    pipeline._fit_light_curves()
