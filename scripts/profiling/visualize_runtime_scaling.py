"""Visualizes profiling results from the `lc_runtime_scaling`` job"""

from pathlib import Path
from pstats import Stats

from matplotlib import pyplot as plt

results_dir = Path(__file__).resolve().parent / 'lc_runtime_scaling'

num_lc = []
runtime = []
for stat_file in results_dir.glob('*.pstat'):
    stats = Stats(str(stat_file))
    stats.calc_callees()

    runtime.append(stats.total_tt)
    num_lc.append(int(stat_file.stem.split('_')[-1]))

fig, axis = plt.subplots()

axis.scatter(num_lc, runtime)
axis.set_xlabel('Number of Light-curves')
axis.set_ylabel('Runtime (s)')
plt.show()
