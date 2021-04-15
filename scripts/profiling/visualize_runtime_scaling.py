"""Visualizes profiling results from the `lc_runtime_scaling`` job"""

from pathlib import Path
from pstats import Stats

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from snat_sim.plasticc import PLAsTICC

results_dir = Path(__file__).resolve().parent / 'lc_runtime_scaling'

num_lc = []
runtime = []
for stat_file in results_dir.glob('*.pstat'):
    stats = Stats(str(stat_file))
    stats.calc_callees()

    runtime.append(stats.total_tt)
    num_lc.append(int(stat_file.stem.split('_')[-1]))

fit_params = np.polyfit(y=runtime, x=num_lc, deg=1)
linear_fit = np.poly1d(fit_params)

cadence_used = 'alt_sched'
total_cadence_lc = PLAsTICC(cadence_used, 11).count_light_curves()
total_cadence = total_cadence_lc * fit_params[0] * u.s

fig, axis = plt.subplots()
axis.scatter(num_lc, runtime)
axis.plot(axis.get_xlim(), linear_fit(axis.get_xlim()), label=f'y={fit_params[0]: .3f} x + {fit_params[1]: .2f}')

axis.set_xlabel('Number of Light-curves')
axis.set_ylabel('Runtime (s)')
axis.set_title(f'Estimated {cadence_used} runtime: {total_cadence.to(u.day): .2f} ({total_cadence_lc} LC)')

axis.legend()
plt.show()
