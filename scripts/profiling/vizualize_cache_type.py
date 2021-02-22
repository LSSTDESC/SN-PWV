"""Visualizes profiling results from the `cache_type_profiling`` job"""

from pathlib import Path
from pstats import Stats

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

results_dir = Path(__file__).resolve().parent / 'cache_type_profiling'
cache_files = ('cache_0.pstat', 'cache_1.pstat', 'cache_2.pstat', 'cache_second_2.pstat')
function_names = ['total_runtime', 'snat_sim/pipeline/plasticc_io/action', 'snat_sim/pipeline/lc_simultion/action', 'snat_sim/pipeline/lc_fitting/action', 'snat_sim/models/calc_airmass']


def load_profile_results(path: Path) -> dict:
    profile = Stats(str(path))
    profile.calc_callees()

    out_dict = {'total_runtime': profile.total_tt}
    for (module_path, *_, func_name), (*_, total_runtime, _) in profile.stats.items():
        relative_module_path = module_path.split('python3.8/')[-1].split('site-packages/')[-1]
        key = relative_module_path.replace('.py', '/' + func_name)
        out_dict[key] = total_runtime

    return out_dict


if __name__ == '__main__':
    # Load results for each profiling run
    profiles = [load_profile_results(results_dir / fname) for fname in cache_files]
    plot_data = [[prof[f] for f in function_names] for prof in profiles]

    # Plot runtimes on a grid
    fig, ax = plt.subplots()
    norm = colors.Normalize(0, np.max(plot_data))
    c = ax.imshow(plot_data, cmap='Blues', norm=norm)
    plt.colorbar(c).ax.set_ylabel('Runtime (s)', rotation=270, labelpad=20)

    # Annotate runtimes so that each square shows time in seconds
    x = np.arange(len(function_names))
    y = np.arange(len(profiles))
    for i in x:
        for j in y:
            ax.annotate(f'{plot_data[j][i]: .2f}', xy=(i - .3, j), color='k')

    ax.set_xticks(x)
    ax.set_xticklabels(function_names, rotation=-30, ha='left')

    ax.set_yticks(y)
    ax.set_yticklabels([Path(fname).stem for fname in cache_files])

    plt.tight_layout()
    plt.show()
