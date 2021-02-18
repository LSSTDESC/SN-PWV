import sys

from pwv_kpno.defaults import ctio

from snat_sim import models
from snat_sim.pipeline import FittingPipeline


def setup_pipeline(cadence, iter_lim):
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

    fitting_pipeline = FittingPipeline(
        cadence=cadence,
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
    fitting_pipeline.load_plastic.num_processes = 0
    fitting_pipeline.simulate_light_curves.num_processes = 0
    fitting_pipeline.fit_light_curves.num_processes = 0
    fitting_pipeline.fits_to_disk.num_processes = 0
    return fitting_pipeline


if __name__ == '__main__':
    try:
        CADENCE = sys.argv[1]

    except IndexError:
        raise ValueError('Cadence not specified on command line.')

    try:
        ITER_LIM = sys.argv[2]

    except IndexError:
        ITER_LIM = 10

    pipeline = setup_pipeline(CADENCE, ITER_LIM)

    print('Loading data')
    pipeline.load_plastic.execute()

    print('Simulating data')
    pipeline.simulate_light_curves.execute()

    print('Fitting data')
    pipeline.fit_light_curves.execute()
