from pwv_kpno.defaults import ctio

from snat_sim import models
from snat_sim.pipeline import FittingPipeline


def setup_pipeline(iter_lim=1500):
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

    pipeline = FittingPipeline(
        cadence='alt_sched',
        sim_model=sn_model_sim,
        fit_model=sn_model_fit,
        vparams=['x0', 'x1', 'c'],
        simulation_pool=4,
        fitting_pool=6,
        out_path='./test.csv',
        iter_lim=iter_lim,
        max_queue=iter_lim + 1  # Allow all data to be loaded so we can run sequentially
    )

    # Run the pipeline in the main thread
    pipeline.load_plastic.num_processes = 0
    pipeline.simulate_light_curves.num_processes = 0
    pipeline.fit_light_curves.num_processes = 0
    pipeline.fits_to_disk.num_processes = 0
    return pipeline


if __name__ == '__main__':
    pipeline = setup_pipeline()

    print('Loading data')
    pipeline.load_plastic.execute()

    print('Simulating data')
    pipeline.simulate_light_curves.execute()

    print('Fitting data')
    pipeline.fit_light_curves.execute()
