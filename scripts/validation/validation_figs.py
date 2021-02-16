"""A Bokeh based script for visualizing pipeline output data from validation runs

To run this script:
    ``bokeh serve [PAH TO SCRIPT] --args [PATH TO PIPELINE CSV OUTPUT] [PATH TO PIPELINE HDF5 OUTPUT]``

To run the validation fits that this script is intended to visualize, see
``validation_fits.sh``.
"""

import sys
from pathlib import Path
from typing import *

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9 as wmap9
from bokeh import layouts, plotting
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Div, Select, Span, Whisker
from bokeh.models.widgets import DataTable, TableColumn
from pwv_kpno.defaults import ctio

import bokeh_acce0ssor
from snat_sim.constants import betoule_cosmo
from snat_sim.models import PWVModel, SNModel, SeasonalPWVTrans, StaticPWVTrans, VariablePWVTrans

GridLayoutType = List[List[plotting.Figure]]


def load_light_curve_sims(path: Path) -> pd.DataFrame:
    """Load a combined table of all light-curve simulation from a directory

    Args:
        path: File path in HDF5 format

    Returns:
        A vertically stacked copy of all light-curve data points
    """

    dataframes = []
    file = h5py.File(path)
    for key in file.keys():
        df = pd.DataFrame(np.array(file[key]))
        df['snid'] = key
        dataframes.append(df)

    data = pd.concat(dataframes, axis=0).set_index('snid')
    data['band'] = data.band.str.decode('utf-8')
    data['zpsys'] = data.zpsys.str.decode('utf-8')
    return data


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

lc_sims = load_light_curve_sims(Path(sys.argv[2]).resolve())
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
pwv_hist = lc_sims.snat_sim_bokeh.histogram('pwv', 'Simulated PWV', line_color='#3A5785', bins=np.arange(0, 35),
                                            **hist_args)
model_hist = lc_sims.snat_sim_bokeh.histogram('pwv_model', 'Modeled PWV', line_color='orange', bins=np.arange(0, 35),
                                              **hist_args)
airmass_hist = lc_sims.snat_sim_bokeh.histogram('airmass', 'Airmass', line_color='grey', bins=np.arange(1, 1.75, .01),
                                                **hist_args)

#  Plot the apparent mag and distance modulus
mag_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mb', 'z', 'Apparent B-band', plot_width=600)
mu_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mu', 'z', 'Fitted Distance Modulus', plot_width=600)

z = np.arange(pipeline_output.sim_z.min(), pipeline_output.sim_z.max() + .005, .005)
mu_scatter.line(z, betoule_cosmo.distmod(z), color='red', legend_label='Betoule et al. 2014')
mu_scatter.line(z, wmap9.distmod(z), color='grey', legend_label='WMAP9')
mu_scatter.legend.click_policy = 'hide'

# Plot the simulated light-curves and their fits
bands = 'ugrizy'
sources = [ColumnDataSource(data=dict(time=[], flux=[], fitted_flux=[], lower_err=[], upper_err=[])) for _ in bands]
colors = ('blue', 'orange', 'green', 'red', 'purple', 'black')

lc_figs = []
for source, color, band in zip(sources, colors, bands):
    lc_plot = plotting.figure(plot_height=400, plot_width=400, title="", toolbar_location=None)
    lc_plot.renderers.append(Span(location=0, dimension='width', line_color='grey', line_dash='dotted', line_width=1))
    lc_plot.add_layout(
        Whisker(source=source, base='time', upper='upper_err', lower='lower_err', line_alpha=.5)
        # Error bars on flux values
    )
    lc_plot.circle(x='time', y='flux', source=source, color=color, legend_label=band)
    lc_plot.line(x='time', y='fitted_flux', source=source, color=color, legend_label=f'Fitted {band}', alpha=.5)
    lc_plot.legend.click_policy = 'hide'

    lc_plot.xaxis.axis_label = 'Time (MJD)'
    lc_plot.yaxis.axis_label = 'Flux'
    lc_figs.append(lc_plot)

# Add the ability to refine plotted light_curves and tabular data per SNID
select_snid = Select(title="SNID", value=pipeline_output.index[0], options=sorted(pipeline_output.index.unique()))

# Add table of fitted parameters
param_table_source = ColumnDataSource(pipeline_output.iloc[:1])
ignore_cols = ['fit_pwv', 'sim_pwv', 'err_pwv', 'message', 'mb', 'abs_mag', 'mu']
param_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in pipeline_output.columns if Ci not in ignore_cols],
    source=param_table_source,
    width=1200,
    height=100
)

# Here we create a tabular representation of the light-curve data
data_table_source = ColumnDataSource(lc_sims.head())
data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in lc_sims.columns],
    source=data_table_source,
    width=1200
)


def update():
    """Call back for updating light-curve plots to reflect the selected SNID"""

    df = lc_sims[lc_sims.index == select_snid.value]
    data_table_source.data = df

    # Update supernova model with fitted parameters so we can plot fitted flux
    fitted_param_values = pipeline_output.loc[select_snid.value]
    sn_model.update({p: fitted_param_values[f'fit_{p}'] for p in sn_model.param_names})

    # Update table of pipeline results
    fitted_param_values = fitted_param_values.fillna('')
    param_table_source.data = {c: [fitted_param_values[c]] for c in pipeline_output.columns if c not in ignore_cols}

    for band, s in zip('ugrizy', sources):
        full_band_name = f'lsst_hardware_{band}'
        band_data = df[df.band == full_band_name]
        s.data = dict(
            time=band_data.time,
            flux=band_data.flux,
            fitted_flux=sn_model.bandflux(full_band_name, band_data.time, zp=band_data.zp, zpsys=band_data.zpsys),
            upper_err=band_data.flux + band_data.fluxerr,
            lower_err=band_data.flux - band_data.fluxerr
        )


select_snid.on_change('value', lambda attr, old, new: update())
update()  # initial load of the data

##############################################################################
# Layout document
##############################################################################

doc_layout = layouts.layout([
    [Div(text=rf'<h1>{validation_path.stem}</h1>')],
    [param_scatter],
    layouts.gridplot([[sim_contour, fit_contour]]),
    layouts.gridplot([[mag_scatter, mu_scatter]]),
    layouts.gridplot([[pwv_hist, pwv_scatter]]),
    layouts.gridplot([[model_hist, model_scatter]]),
    layouts.gridplot([[airmass_hist, airmass_scatter]]),
    [select_snid],
    [param_table],
    layouts.gridplot(np.reshape(lc_figs, (2, 3)).tolist()),
    [data_table]
])

curdoc().add_root(doc_layout)
