"""Visualizes profiling results from the `cache_type_profiling`` job"""

from pathlib import Path
from pstats import Stats

import numpy as np
from matplotlib import pyplot as plt


def read_runtime(path: Path) -> dict:
    profile = Stats(str(path))
    profile.calc_callees()
    return profile.total_tt


def load_results_dir(path: Path) -> list:
    cache_files = ('cache_0.pstat', 'cache_1.pstat', 'cache_2.pstat', 'cache_second_2.pstat')
    return [read_runtime(path / fname) for fname in cache_files]


def make_bar_plot(results_dir, runs):
    width = .9 / len(runs)
    x_vals = np.arange(4)
    _, ax = plt.subplots()

    for i, num_lc in enumerate(runs):
        x_vals_temp = x_vals + i * width
        runtimes = load_results_dir(results_dir / f'run_{num_lc}')
        ax.bar(x_vals_temp, runtimes, width, align='center', label=f'{num_lc}')
        for x, y in zip(x_vals_temp, runtimes):
            ax.text(x, y + 5, f'{y: .1f}', ha='center', va='bottom')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(('None', 'Memoize', 'Joblib', 'Joblib Cached'))
    ax.set_ylabel('Runtime (s)')
    ax.set_xlabel('Caching Method')
    ax.set_ylim(0, 300)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _results_dir = Path(__file__).resolve().parent / 'cache_type_profiling'
    _runs = (500, 1000, 2000)
    make_bar_plot(_results_dir, _runs)
