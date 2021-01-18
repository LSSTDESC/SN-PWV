import sys
from pathlib import Path
from typing import *

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
print(Path(__file__).parent.parent.parent)

import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9 as wmap9
from astropy.table import Table, vstack
from bokeh import layouts, plotting
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Select, Span
from bokeh.models.widgets import DataTable, TableColumn
from pwv_kpno.defaults import ctio

import bokeh_accessor
from snat_sim.constants import betoule_cosmo
from snat_sim.models import PWVModel, SNModel, SeasonalPWVTrans, StaticPWVTrans, VariablePWVTrans

GridLayoutType = List[List[plotting.Figure]]


def load_light_curve_sims(directory: Path) -> pd.DataFrame:
    """Load a combined table of all light-curve simulation from a directory

    Args:
        directory: A directory of light-curves in ecsv format

    Returns:
        A vertically stacked copy of all light-curve data points
    """

    light_curves = []
    for path in directory.glob('*.ecsv'):
        data = Table.read(path)
        data['snid'] = path.stem
        light_curves.append(data)

    all_data = vstack(light_curves, metadata_conflicts='silent').to_pandas(index='snid')
    all_data.index = all_data.index.astype(str)
    return all_data


def load_pipeline_output(path: Path) -> pd.DataFrame:
    """Read a pipeline output file

    Args:
        path: File path to read (in csv format)

    Returns:
        A copy of the pipeline output data failed fits dropped
    """

    df = pd.read_csv(path, index_col=0).replace(-99.99, np.nan).dropna(subset=['chisq'])
    df.index = df.index.astype(str)
    return df


def build_sn_model(variability, pwv_model, source='salt2-extended'):
    if variability.isnumeric():
        transmission_effect = StaticPWVTrans()
        transmission_effect.set(pwv=float(variability))
        effect = transmission_effect

    elif variability == 'epoch':
        effect = VariablePWVTrans(pwv_model)

    elif variability == 'seasonal':
        effect = SeasonalPWVTrans.from_pwv_model(pwv_model)

    else:
        raise NotImplementedError(f'Unknown variability: {variability}')

    return SNModel(
        source,
        effects=[effect],
        effect_names=[''],
        effect_frames=['obs'])


##############################################################################
# Load pipeline data
##############################################################################

# Build a copy of the SN and PWV models used in the pipeline
validation_path = Path(sys.argv[1]).resolve()
pwv_model = PWVModel.from_suominet_receiver(ctio, 2016, [2017])  # PWV used in simulation
sn_model = build_sn_model(validation_path.stem.split('_')[-1], pwv_model)

pipeline_output = load_pipeline_output(validation_path)
pipeline_output['mu'] = pipeline_output['mb'] - pipeline_output['abs_mag']

lc_sims = load_light_curve_sims(validation_path.parent / validation_path.stem)
lc_sims['pwv_model'] = pwv_model.pwv_zenith(lc_sims['time'])

##############################################################################
# Build plots
##############################################################################

# Corner plot of fitted vs simulated parameters
params = ('z', 'x0', 'x1', 'c')
param_scatter = pipeline_output.snat_sim_bokeh.corner(
    x_vals=[f'sim_{p}' for p in params],
    y_vals=[f'fit_{p}' for p in params],
    x_labels=[f'Simulated {p}' for p in params],
    y_labels=[f'Fitted {p}' for p in params],
    size=5
)

# Stretch color contour plots
sim_contour, fit_contour = pipeline_output.snat_sim_bokeh.scatter(
    x_vals=['sim_c', 'fit_c'], y_vals=['sim_x1', 'fit_x1'],
    x_labels=['Simulated c', 'Fitted c'],
    y_labels=['Simulated x1', 'Fitted x1'],
    contour=True)

# Scatter plots of PWV values and airmasses
scat_args = dict(plot_height=400, plot_width=800, alpha=.1)
pwv_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'pwv', 'Time', 'Simulated PWV', **scat_args)
model_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'pwv_model', 'Time', 'Modeled WPV', color='orange', **scat_args)
airmass_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'airmass', 'Time', 'Airmass', color='grey', **scat_args)

# Histograms of PWV values and airmasses
hist_args = dict(plot_width=400, color='white')
pwv_hist = lc_sims.snat_sim_bokeh.histogram('pwv', 'Simulated PWV', line_color='#3A5785', bins=np.arange(0, 35), **hist_args)
model_hist = lc_sims.snat_sim_bokeh.histogram('pwv_model', 'Modeled PWV', line_color='orange', bins=np.arange(0, 35), **hist_args)
airmass_hist = lc_sims.snat_sim_bokeh.histogram('airmass', 'Airmass', line_color='grey', bins=np.arange(1, 1.75, .01), **hist_args)

#  Plot the apparent mag and distance modulus
mag_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mb', 'z', 'Apparent B-band', plot_width=600)
mu_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mu', 'z', 'Fitted Distance Modulus', plot_width=600)

z = np.arange(pipeline_output.sim_z.min(), pipeline_output.sim_z.max() + .005, .005)
mu_scatter.line(z, betoule_cosmo.distmod(z), color='red', legend_label='Betoule et al. 2014')
mu_scatter.line(z, wmap9.distmod(z), color='grey', legend_label='WMAP9')
mu_scatter.legend.click_policy = 'hide'

# The following code plots light-curve fits on a per SNID basis
bands = 'ugrizy'
sources = [ColumnDataSource(data=dict(time=[], flux=[], fitted_flux=[])) for _ in bands]
colors = ('blue', 'orange', 'green', 'red', 'purple', 'black')

lc_figs = []
for source, color, band in zip(sources, colors, bands):
    lc_plot = plotting.figure(plot_height=400, plot_width=400, title="", toolbar_location=None)
    lc_plot.renderers.append(Span(location=0, dimension='width', line_color='grey', line_width=1))
    lc_plot.circle(x='time', y='flux', source=source, color=color, legend_label=band)
    lc_plot.line(x='time', y='fitted_flux', source=source, color=color, legend_label=f'Fitted {band}', alpha=.5)
    lc_figs.append(lc_plot)

# Here we create a tabular representation of the light-curve data
table_source = ColumnDataSource(lc_sims.head())
data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in lc_sims.columns],
    source=table_source,
    width=1200
)

# Add the ability to refine plotted light_curves and tabular data per SNID
select_snid = Select(title="SNID", value=lc_sims.index[0], options=sorted(lc_sims.index.unique()))


def update():
    """Call back for updating light-curve plots to reflect the selected SNID"""

    df = lc_sims[lc_sims.index == select_snid.value]
    table_source.data = df

    fitted_param_values = pipeline_output.loc[select_snid.value]
    sn_model.update({p: fitted_param_values[f'fit_{p}'] for p in sn_model.param_names})

    for band, s in zip('ugrizy', sources):
        full_band_name = f'lsst_hardware_{band}'
        band_data = df[df.band == full_band_name]
        s.data = dict(
            time=band_data.time,
            flux=band_data.flux,
            fitted_flux=sn_model.bandflux(full_band_name, band_data.time, zp=band_data.zp, zpsys=band_data.zpsys)
        )


select_snid.on_change('value', lambda attr, old, new: update())
update()  # initial load of the data

##############################################################################
# Layout document
##############################################################################

doc_layout = layouts.layout([
    [param_scatter],
    layouts.gridplot([[sim_contour, fit_contour]]),
    layouts.gridplot([[mag_scatter, mu_scatter]]),
    layouts.gridplot([[pwv_hist, pwv_scatter]]),
    layouts.gridplot([[model_hist, model_scatter]]),
    layouts.gridplot([[airmass_hist, airmass_scatter]]),
    [select_snid],
    layouts.gridplot(np.reshape(lc_figs, (2, 3)).tolist()),
    [data_table]
])

curdoc().add_root(doc_layout)
