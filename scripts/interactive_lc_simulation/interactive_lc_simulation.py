"""Interactive Bokeh application for plotting light-curve simulations

Run this script via the ``bokeh`` command line tool:

.. code-block::
   bokeh serve [FILE PATH]
"""

import sys
from copy import copy
from pathlib import Path

import numpy as np
from bokeh import models
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure

_file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_file_dir.parent.parent))

from snat_sim.models import ReferenceCatalog, SNModel, StaticPWVTrans
from tests import mock

# SNCosmo source to use when plotting
SOURCE = 'salt2-extended'
BANDS = tuple(f'lsst_hardware_{b}' for b in 'ugriz')
REFERENCE_CATALOG = ReferenceCatalog('G2', 'M5', 'K2')


class SimulatedParamWidgets:
    """Collection of Bokeh Widgets for specifying simulation parameters"""

    top_section_div = models.Div(text='<h2>Simulated Parameters</h2>')

    # User input widgets for setting model parameters
    sim_z_slider = models.Slider(start=0.001, end=1, value=.55, step=.01, title='z')
    sim_t0_slider = models.Slider(start=-10, end=10, value=-2, step=.01, title='t0')
    sim_x0_slider = models.Slider(start=0.001, end=2, value=.25, step=.01, title='x0')
    sim_x1_slider = models.Slider(start=-1, end=1, value=0.11, step=.01, title='x1')
    sim_c_slider = models.Slider(start=-1, end=1, value=-.05, step=.01, title='c')
    sampling_input = models.TextInput(value='4', title='Sampling (Days):')
    sim_pwv_slider = models.Slider(start=-0, end=15, value=7, step=.1, title='PWV')
    plot_button = models.Button(label='Plot Light-Curve', button_type='success')

    snr_input = models.TextInput(value='10.0', title='SNR:', default_size=220)
    checkbox = models.CheckboxGroup(labels=["Plot SNR", 'Subtract Reference Star'], active=[0])

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
        fit_c_slider,
        plot_model_button,
        fit_button
    ]


class Callbacks(SimulatedParamWidgets, FittedParamWidgets):
    """Assigns callbacks and establishes interactive behavior"""

    plotted_fits = []
    plotted_data = []
    sim_data = None

    def __init__(self, main_figure, spec_figure, fit_results_div, source='salt2-extended'):
        """Assign callbacks to plot widgets

        Args:
            main_figure  (Figure): Bokeh figure to render plots on
            fit_results_div (Div): Used to display fit results as text
            source  (Str, Source): SNCosmo source of the desired model
        """

        # Widgets for plotting / fit results
        self.main_figure = main_figure
        self.spec_figure = spec_figure
        self.fit_results_div = fit_results_div
        self.sn_model_without_pwv = SNModel(source)
        self.sn_model_with_pwv = SNModel(
            source,
            effects=[StaticPWVTrans()],
            effect_frames=['obs'],
            effect_names=[''])

        # Assign callbacks
        self.plot_button.on_click(self.plot_simulated_flux)
        self.fit_button.on_click(self.fit_light_curve)
        self.plot_model_button.on_click(self.plot_current_model)

    def _clear_fitted_lines(self):
        """Remove model fits from the plot"""

        while self.plotted_fits:
            line = self.plotted_fits.pop()
            try:
                self.main_figure.renderers.remove(line)

            except ValueError:
                self.spec_figure.renderers.remove(line)

    def _clear_plotted_object_data(self):
        """Remove simulated light-curve data points from the plot"""

        while self.plotted_data:
            line = self.plotted_data.pop()
            try:
                self.main_figure.renderers.remove(line)

            except ValueError:
                self.spec_figure.renderers.remove(line)

    def plot_simulated_flux(self, event=None):
        """Simulate and plot a light-curve"""

        # Clear the plot
        self._clear_plotted_object_data()
        self._clear_fitted_lines()
        params = dict(
            z=self.sim_z_slider.value,
            t0=self.sim_t0_slider.value,
            x0=self.sim_x0_slider.value,
            x1=self.sim_x1_slider.value,
            c=self.sim_c_slider.value,
            pwv=self.sim_pwv_slider.value
        )

        # Simulate a light-curve
        obs = mock.create_mock_cadence(np.arange(-10, 51, float(self.sampling_input.value)), BANDS)
        self.sn_model_with_pwv.update(params)
        self.sim_data = self.sn_model_with_pwv.simulate_lc(obs, fixed_snr=float(self.snr_input.value), scatter=False)

        # Scale flux by reference star
        if 1 in self.checkbox.active:
            self.sim_data = REFERENCE_CATALOG.calibrate_lc(
                self.sim_data, self.sim_pwv_slider.value
            )

        # Update the main plot with simulated flux data
        sim_as_astropy = self.sim_data.to_astropy()
        for band, color in zip(BANDS, palette):
            band_data = sim_as_astropy[sim_as_astropy['band'] == band]
            x = band_data['time']
            y = band_data['flux']
            yerr = band_data['fluxerr']

            circ = self.main_figure.circle(x=x, y=y, color=color, legend_label=band)
            self.plotted_data.append(circ)

            if 0 in self.checkbox.active:
                err_bar = self.main_figure.multi_line(
                    np.transpose([x, x]).tolist(),
                    np.transpose([y - yerr, y + yerr]).tolist(),
                    color=color)

                self.plotted_data.append(err_bar)

        # Update plot of simulated spectrum
        wave = np.arange(self.sn_model_with_pwv.minwave(), self.sn_model_with_pwv.maxwave())
        spec_line = self.spec_figure.line(x=wave, y=self.sn_model_with_pwv.flux(0, wave), legend_label='Observed')
        self.plotted_data.append(spec_line)

        # Match fitted param sliders to sim param sliders
        self.fit_t0_slider.update(value=self.sim_t0_slider.value)
        self.fit_x0_slider.update(value=self.sim_x0_slider.value)
        self.fit_x1_slider.update(value=self.sim_x1_slider.value)
        self.fit_c_slider.update(value=self.sim_c_slider.value)

    def fit_light_curve(self, event=None):
        """Fit the simulated light-curve and plot the results"""

        self.sn_model_without_pwv.set(
            z=self.sim_z_slider.value,
            t0=self.fit_t0_slider.value,
            x0=self.fit_x0_slider.value,
            x1=self.fit_x1_slider.value,
            c=self.fit_c_slider.value
        )

        try:
            result = self.sn_model_without_pwv.fit_lc(
                self.sim_data,
                vparam_names=['t0', 'x0', 'x1', 'c'])

        except Exception as e:
            self.fit_results_div.update(text=str(e))
            return

        else:
            fitted_model = copy(self.sn_model_without_pwv)
            fitted_model.update(dict(zip(result.param_names, result.parameters)))

        # Set fitted param sliders to reflect the fitted parameters
        self.fit_t0_slider.update(value=fitted_model['t0'])
        self.fit_x0_slider.update(value=fitted_model['x0'])
        self.fit_x1_slider.update(value=fitted_model['x1'])
        self.fit_c_slider.update(value=fitted_model['c'])
        self.plot_current_model()

        # Update results div
        keys = 'message', 'ncall', 'chisq', 'ndof', 'vparam_names', 'param_names', 'parameters'
        text = '<h4>Fit Results</h4>'
        text += '<br>'.join(f'{k}: {result[k]}' for k in keys)

        text += '<br><h4>Sim Mag</h4>'
        text += f'standard::b (AB): {self.sn_model_with_pwv.source_peakmag("standard::b", "AB")}'
        text += f'<br>peak standard::b (AB): {self.sn_model_with_pwv.source_peakabsmag("standard::b", "AB")}'

        text += '<br><h4>Fitted Mag</h4>'
        text += f'standard::b (AB): {fitted_model.source_peakmag("standard::b", "AB")}'
        text += f'<br>peak standard::b (AB): {fitted_model.source_peakabsmag("standard::b", "AB")}'
        self.fit_results_div.update(text=text)

    def plot_current_model(self, event=None):
        """Plot the model (without PWV) using the initial guess parameters"""

        self.sn_model_without_pwv.set(
            z=self.sim_z_slider.value,
            t0=self.fit_t0_slider.value,
            x0=self.fit_x0_slider.value,
            x1=self.fit_x1_slider.value,
            c=self.fit_c_slider.value
        )

        self._clear_fitted_lines()
        self.fit_results_div.update(text='')
        time_arr = np.arange(-25, 55, .5)
        for band, color in zip(BANDS, palette):
            line = self.main_figure.line(
                x=time_arr,
                y=self.sn_model_without_pwv.bandflux(band, time_arr, zp=30, zpsys='ab'),
                color=color
            )

            self.plotted_fits.append(line)

        wave = np.arange(self.sn_model_without_pwv.minwave(), min(self.sn_model_without_pwv.maxwave(), 12000))
        spec_line = self.spec_figure.line(x=wave, y=self.sn_model_without_pwv.flux(0, wave), color='red', legend_label='Model')
        self.plotted_fits.append(spec_line)


###############################################################################
# Assign Callbacks
###############################################################################

# The figure objects to plot on
central_figure = figure(plot_height=400, plot_width=700, sizing_mode='scale_both', toolbar_location='above')
spectrum_figure = figure(plot_height=200, plot_width=700, sizing_mode='scale_both')

# Div for displaying fitted ``Results`` object
fit_results = models.Div()
callbacks = Callbacks(central_figure, spectrum_figure, fit_results, SOURCE)

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
    central_figure,
    spectrum_figure,
    fit_results,
    height=1200,
    sizing_mode="scale_width")

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
