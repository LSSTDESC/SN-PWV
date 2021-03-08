from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=(7, 2),
    height_ratios=(2, 7, 2),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# residuals_ax = fig.add_subplot(gs[2, 0], sharex=ax)

data = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'snat_sim_runs' / 'snat_sim.alt_sched_rolling.csv')
data = data[data.mb > 0]  # Drop results that are masked with -99

# Rename data so that the minimization accessor can identify the correct values
# minimizer_data = data.rename(columns={f'fit_{p}': p for p in ('t0', 'z', 'x0', 'x1', 'c')})
# min_res = minimizer_data.head().snat_sim.minimize(
#     H0=sc.betoule_H0,
#     Om0=sc.betoule_omega_m,
#     w0=-1,
#     fix_w0=True,
#     abs_mag=sc.betoule_abs_mb,
#     fix_abs_mag=True,
#     alpha=sc.betoule_alpha,
#     beta=sc.betoule_beta,
# )

x, y = data.fit_z, data.mb

ax.scatter(x, y, s=1, label='alt_sched_rolling')
ax.tick_params(axis="x", labelbottom=False)
ax.set_ylabel(r'Fitted $m_B$')
ax.set_xlabel('Redshift')
ax.set_xlim(0, 0.9)
ax.legend(loc='lower right')

ax_histx.hist(x, bins=np.arange(0, 1, .025))
ax_histx.tick_params(axis="x", labelbottom=False)

ax_histy.hist(y, bins=np.arange(10, 30, 0.5), orientation='horizontal')
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.tick_params(axis="x", rotation=270)

# residuals_ax.set_xlabel('Redshift')
# residuals_ax.set_ylabel('Residuals')

plt.show()
