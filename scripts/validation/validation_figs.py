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
from bokeh.models import ColumnDataSource, Select
from pwv_kpno.defaults import ctio

import bokeh_accessor
from snat_sim._data_paths import data_paths
from snat_sim.constants import betoule_cosmo
from snat_sim.models import PWVModel

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


def build_validation_page(
        validation_path,
        pwv_model,
        params=('z', 'x0', 'x1', 'c'),
        contours=(['sim_c', 'fit_c'], ['sim_x1', 'fit_x1'])
):
    pipeline_output = load_pipeline_output(validation_path)
    pipeline_output['mu'] = pipeline_output['mb'] - pipeline_output['abs_mag']

    lc_sims = load_light_curve_sims(validation_path.parent / validation_path.stem)
    lc_sims['pwv_model'] = pwv_model.pwv_zenith(lc_sims['time'])

    param_scatter = pipeline_output.snat_sim_bokeh.corner(
        x_vals=[f'sim_{p}' for p in params],
        y_vals=[f'fit_{p}' for p in params],
        x_labels=[f'Simulated {p}' for p in params],
        y_labels=[f'Fitted {p}' for p in params],
        size=5
    )

    param_contour = pipeline_output.snat_sim_bokeh.scatter(x_vals=contours[0], y_vals=contours[1], contour=True)

    scat_args = dict(plot_height=400, plot_width=800, alpha=.1)
    pwv_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'pwv', 'Time', 'PWV', **scat_args)
    model_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'pwv_model', 'Time', 'Model', color='orange', **scat_args)
    airmass_scatter = lc_sims.snat_sim_bokeh.scatter('time', 'airmass', 'Time', 'Airmass', color='grey', **scat_args)

    hist_args = dict(plot_width=400, color='white')
    pwv_hist = lc_sims.snat_sim_bokeh.histogram('pwv', line_color='#3A5785', bins=np.arange(0, 35), **hist_args)
    model_hist = lc_sims.snat_sim_bokeh.histogram('pwv_model', line_color='orange', bins=np.arange(0, 35), **hist_args)
    airmass_hist = lc_sims.snat_sim_bokeh.histogram('airmass', line_color='grey', bins=np.arange(1, 1.75, .01), **hist_args)

    mag_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mb', 'z', 'Apparent B-band', plot_width=600)
    mu_scatter = pipeline_output.snat_sim_bokeh.scatter('sim_z', 'mu', 'z', 'Fitted Distance Modulus', plot_width=600)

    z = np.arange(pipeline_output.sim_z.min(), pipeline_output.sim_z.max() + .005, .005)
    mu_fig = mu_scatter.children[-1].children[0][0]
    mu_fig.line(z, betoule_cosmo.distmod(z), color='red', legend_label='Betoule et al. 2014')
    mu_fig.line(z, wmap9.distmod(z), color='grey', legend_label='WMAP9')
    mu_fig.legend.click_policy = 'hide'

    bands = 'ugrizy'
    sources = [ColumnDataSource(data=dict(time=[], flux=[])) for _ in bands]
    colors = ('blue', 'orange', 'green', 'red', 'purple', 'black')

    lc_figs = []
    for source, color, band in zip(sources, colors, bands):
        lc_plot = plotting.figure(plot_height=400, plot_width=400, title="", toolbar_location=None)
        lc_plot.circle(x='time', y='flux', source=source, color=color, legend_label=band)
        lc_figs.append(lc_plot)

    select_snid = Select(title="SNID", options=sorted(lc_sims.index.unique()))

    def update():
        df = lc_sims[lc_sims.index == select_snid.value]
        for band, source in zip('ugrizy', sources):
            band_data = df[df.band == f'lsst_hardware_{band}']
            source.data = dict(time=band_data.time, flux=band_data.flux)

    select_snid.on_change('value', lambda attr, old, new: update())

    doc_layout = layouts.layout([
        [param_scatter],
        [param_contour],
        [mag_scatter, mu_scatter],
        [pwv_hist, pwv_scatter],
        [model_hist, model_scatter],
        [airmass_hist, airmass_scatter],
        [select_snid],
        np.reshape(lc_figs, (3, 2)).tolist()

    ])

    update()  # initial load of the data
    return doc_layout


layout = build_validation_page(
    data_paths.data_dir / 'validation' / 'pwv_sim_epoch_fit_epoch.csv',
    PWVModel.from_suominet_receiver(ctio, 2016, [2017]))

curdoc().add_root(layout)
