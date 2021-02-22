"""Visualizes profiling results from the `core_allocation_scan`` job"""

from pathlib import Path
from pstats import Stats

import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import pyplot as plt

results_dir = Path(__file__).resolve().parent / 'core_allocation_scan'

profiling_data = []
for batch_of_profiling_runs in results_dir.glob('*'):
    if not batch_of_profiling_runs.is_dir():
        continue

    run_times_for_batch = dict()
    for stat_file in batch_of_profiling_runs.glob('*.pstat'):
        allocated_cores = int(stat_file.stem.split('_')[-1])

        stats = Stats(str(stat_file))
        stats.calc_callees()
        time = stats.total_tt

        run_times_for_batch[allocated_cores] = time

    profiling_data.append(run_times_for_batch)

plot_data = pd.DataFrame(profiling_data)
reference_val = plot_data[min(plot_data.columns)]
plot_data = ((plot_data.transpose() - reference_val) / reference_val).transpose()
median = plot_data.median()
std = plot_data.std()

fig, axis = plt.subplots()
axis.scatter(median.index, median)
axis.errorbar(median.index, median, yerr=std, linestyle='')

xmin, xmax = median.index.min() - 2, median.index.max() + 2
axis.set_xlim(xmin, xmax)
axis.set_xlabel('Forked Fitting Processes')
axis.xaxis.set_major_locator(ticker.MultipleLocator(2))
axis.xaxis.set_minor_locator(ticker.MultipleLocator(1))

twiny = plt.twiny()
twiny.set_xlim(62 - xmin, 62 - xmax)
twiny.set_xlabel('Forked Simulation Processes')
twiny.xaxis.set_major_locator(ticker.MultipleLocator(2))
twiny.xaxis.set_minor_locator(ticker.MultipleLocator(1))

axis.set_ylabel('Median runtime improvement for 250 LCs (%)')
plt.show()
