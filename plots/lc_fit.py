# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Interactive plotting for light-curve fits"""

import sys
from pathlib import Path

import numpy as np
import sncosmo
from astropy.table import Table, vstack
from bokeh import models
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure

_file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_file_dir.parent))

from sn_analysis.modeling import get_model_with_pwv
from sn_analysis.utils import register_decam_filters
from sn_analysis.reference import subtract_ref_from_lc

# SNCosmo source to use when plotting
SOURCE = 'salt2-extended'
BANDS = tuple(f'decam_{b}' for b in 'riz')
register_decam_filters(force=True)


class SimulatedParamWidgets:
    """Collection of Bokeh Widgets for specifying simulation parameters"""

    top_section_div = models.Div(text='<h2>Simulated Parameters</h2>')

    # User input widgets for setting model parameters
    sim_z_slider = models.Slider(start=0.001, end=1, value=.1, step=.001, title='z')
    sim_t0_slider = models.Slider(start=-10, end=10, value=0, step=.001, title='t0')
    sim_x0_slider = models.Slider(start=0.001, end=2, value=.1, step=.001, title='x0')
    sim_x1_slider = models.Slider(start=-1, end=1, value=0, step=.001, title='x1')
    sim_c_slider = models.Slider(start=-1, end=1, value=0, step=.001, title='c')
    sampling_input = models.TextInput(value='1', title='Sampling (Days):')
    sim_pwv_slider = models.Slider(start=-0, end=15, value=4, step=.1, title='PWV')
    plot_button = models.Button(label='Plot Light-Curve', button_type='success')

    snr_input = models.TextInput(value='0.1', title='SNR:', default_size=220)
    checkbox = models.CheckboxGroup(labels=["Plot SNR", 'Subtract Reference Star'])

    # Having all inputs as a list is useful when constructing layouts
    # as it establishes the default column order
    sim_params_widgets_list = [
        top_section_div,
        sim_z_slider,
        sim_t0_slider,
        sim_x0_slider,
        sim_x1_slider,
        sim_c_slider,
        sim_pwv_slider,
        sampling_input,
        snr_input,
        checkbox,
        plot_button
    ]


class FittedParamWidgets:
    """Collection of Bokeh Widgets for fitting a light-curve parameters"""

    top_section_div = models.Div(text=(
        '<h2>Fitted Parameters</h2>'
        '<p>Current parameter values are used as an initial guess when fitting. '
        'Note that redshift values are not varied as part of the fit.</p>'
    ))

    # User input widgets for setting model parameters
    fit_t0_slider = models.Slider(start=-10, end=10, value=0, step=1E-4, title='t0')
    fit_x0_slider = models.Slider(start=0.001, end=2, value=.1, step=1E-4, title='x0')
    fit_x1_slider = models.Slider(start=-1, end=1, value=0, step=1E-4, title='x1')
    fit_c_slider = models.Slider(start=-1, end=1, value=0, step=1E-4, title='c')
    plot_model_button = models.Button(label='Plot Model', button_type='warning')
    fit_button = models.Button(label='Fit Light-Curve', button_type='success')

    fitted_params_widgets_list = [
        top_section_div,
        fit_t0_slider,
        fit_x0_slider,
        fit_x1_slider,
        fit_c_slider, plot_model_button,
        fit_button
    ]


class Simulation(SimulatedParamWidgets, FittedParamWidgets):

    def __init__(self, source='salt2-extended'):
        self.source = source
        self._sim_data = None

    def simulate_lc(self):
        # Instantiate a model with simulation params from input widgets
        model = get_model_with_pwv(self.source)

        model.update(dict(
            z=self.sim_z_slider.value,
            t0=self.sim_t0_slider.value,
            x0=self.sim_x0_slider.value,
            x1=self.sim_x1_slider.value,
            c=self.sim_c_slider.value,
            pwv=self.sim_pwv_slider.value
        ))

        # Simulate magnitude in each band
        time_arr = np.arange(-20, 50, float(self.sampling_input.value))
        band_sim_data = []
        for band, color in zip(BANDS, palette):
            flux = model.bandflux(band, time_arr, zpsys='ab', zp=25)
            band_data = Table([time_arr, flux], names=['time', 'flux'])
            band_data['band'] = band
            band_sim_data.append(band_data)

        # Save the data so it can be fitted later on
        lc_data = vstack(band_sim_data)
        lc_data['fluxerr'] = float(self.snr_input.value) * lc_data['flux']
        lc_data['zpsys'] = 'ab'
        lc_data['zp'] = 25

        if 1 in self.checkbox.active:
            lc_data = subtract_ref_from_lc(lc_data, self.sim_pwv_slider.value)

        return lc_data


class Callbacks(Simulation):
    """Assigns callbacks and establishes interactive behavior"""

    plotted_fits = []
    plotted_data = []
    sim_data = None

    def __init__(self, figure, fit_results_div, source='salt2-extended'):
        """Assign callbacks to plot widgets

        Args:
            figure       (Figure): Bokeh figure to render plots on
            fit_results_div (Div): Used to display fit results as text
            source  (Str, Source): SNCosmo source of the desired model
        """

        super().__init__(source)

        # Widgets for plotting / fit results
        self.figure = figure
        self.fit_results_div = fit_results_div

        # Assign callbacks
        self.plot_button.on_click(self.plot_light_curve)
        self.fit_button.on_click(self.fit_light_curve)
        self.plot_model_button.on_click(self.plot_current_model)

    def _clear_fitted_lines(self):
        """Remove model fits from the plot"""

        while self.plotted_fits:
            self.figure.renderers.remove(self.plotted_fits.pop())

    def _clear_plotted_object_data(self):
        """Remove simulated light-curve data points from the plot"""

        while self.plotted_data:
            self.figure.renderers.remove(self.plotted_data.pop())

    def plot_light_curve(self, event=None):
        """Simulate and plot a light-curve"""

        # Clear the plot
        self._clear_plotted_object_data()
        self._clear_fitted_lines()

        # Update the plot with simulated data
        self.sim_data = self.simulate_lc()
        for band, color in zip(BANDS, palette):
            band_data = self.sim_data[self.sim_data['band'] == band]
            x = band_data['time']
            y = band_data['flux']
            yerr = band_data['fluxerr']

            circ = self.figure.circle(x=x, y=y, color=color)
            self.plotted_data.append(circ)

            # if 0 in self.checkbox.active[:-1]:
            #     err_bar = self.figure.multi_line(
            #         np.transpose([x, x]),
            #         np.transpose([y - yerr, y + yerr]),
            #         color=color)

            #     self.plotted_data.append(err_bar)

        # Match fitted param sliders to sim param sliders
        self.fit_t0_slider.update(value=self.sim_t0_slider.value)
        self.fit_x0_slider.update(value=self.sim_x0_slider.value)
        self.fit_x1_slider.update(value=self.sim_x1_slider.value)
        self.fit_c_slider.update(value=self.sim_c_slider.value)

    def fit_light_curve(self, event=None):
        """Fit the simulated light-curve and plot the results"""

        model = sncosmo.Model(self.source)
        model.update(dict(
            z=self.sim_z_slider.value,
            t0=self.fit_t0_slider.value,
            x0=self.fit_x0_slider.value,
            x1=self.fit_x1_slider.value,
            c=self.fit_c_slider.value
        ))

        try:
            result, fitted_model = sncosmo.fit_lc(
                self.sim_data,
                model,
                vparam_names=['t0', 'x0', 'x1', 'c'])

        except Exception as e:
            self.fit_results_div.update(text=str(e))
            return

        # Update results div
        keys = 'message', 'ncall', 'chisq', 'ndof', 'vparam_names', 'param_names', 'parameters'
        text = '<h4>Fit Results</h4>'
        text += '<br>'.join(f'{k}: {result[k]}' for k in keys)

        text += '<br><h4>Sim Mag</h4>'
        text += f'standard::b (AB): {model.source_peakmag("standard::b", "AB")}'
        text += f'<br>peak standard::b (AB): {model.source_peakabsmag("standard::b", "AB")}'

        text += '<br><h4>Fitted Mag</h4>'
        text += f'standard::b (AB): {fitted_model.source_peakmag("standard::b", "AB")}'
        text += f'<br>peak standard::b (AB): {fitted_model.source_peakabsmag("standard::b", "AB")}'
        self.fit_results_div.update(text=text)

        # Plot the fitted model
        self._clear_fitted_lines()
        time_arr = np.arange(-25, 55, .5)
        for band, color in zip(BANDS, palette):
            line = self.figure.line(
                x=time_arr,
                y=fitted_model.bandflux(band, time_arr, zp=25, zpsys='ab'),
                color=color,
                # legend_label=band
            )

            self.plotted_fits.append(line)

        # Match fitted param sliders to sim param sliders
        self.fit_t0_slider.update(value=self.sim_t0_slider.value)
        self.fit_x0_slider.update(value=self.sim_x0_slider.value)
        self.fit_x1_slider.update(value=self.sim_x1_slider.value)
        self.fit_c_slider.update(value=self.sim_c_slider.value)

    def plot_current_model(self, event=None):
        """Plot the model using the initial guess parameters"""

        model = sncosmo.Model(self.source)
        model.update(dict(
            z=self.sim_z_slider.value,
            t0=self.fit_t0_slider.value,
            x0=self.fit_x0_slider.value,
            x1=self.fit_x1_slider.value,
            c=self.fit_c_slider.value
        ))

        self._clear_fitted_lines()
        self.fit_results_div.update(text='')
        time_arr = np.arange(-25, 55, .5)
        for band, color in zip(BANDS, palette):
            line = self.figure.line(
                x=time_arr,
                y=model.bandflux(band, time_arr, zp=25, zpsys='ab'),
                color=color,
                # legend_label=band
            )

            self.plotted_fits.append(line)


###############################################################################
# Assign Callbacks
###############################################################################

# The figure to plot on
figure = figure(plot_height=400, plot_width=700, sizing_mode='scale_both', toolbar_location='above')
figure.legend.location = "top_left"
figure.legend.click_policy = "hide"

# Div for displaying fitted ``Results`` object
fit_results_div = models.Div()
callbacks = Callbacks(figure, fit_results_div, SOURCE)

###############################################################################
# Layout the page
###############################################################################

# Descriptive text for the top of the page
header_path = _file_dir / "lc_fit_header.html"
with header_path.open() as infile:
    header_div = models.Div(text=infile.read(), sizing_mode="stretch_width")

left_column = column(
    *callbacks.sim_params_widgets_list,
    width=320,
    height=1000,
    sizing_mode="fixed")

center_column = column(
    figure,
    fit_results_div,
    height=1000, sizing_mode="scale_width")

right_column = column(
    *callbacks.fitted_params_widgets_list,
    width=320,
    height=1000,
    sizing_mode="fixed")

doc_layout = layout(
    [
        [header_div],
        [left_column, center_column, right_column]
    ], sizing_mode="scale_both")

curdoc().add_root(doc_layout)
curdoc().title = "SN Light-Curves"
