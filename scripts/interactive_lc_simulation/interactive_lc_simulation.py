"""Interactive Bokeh application for plotting light-curve fits and simulations

Application widgets are grouped into classes by common purpose
(e.g., simulation inputs, fitted model parameters). Each class provides
methods for formatting widgets into usable layouts (rows, columns, etc.).
Where appropriate, convenience methods are provided for manipulating widget
states in bulk.
"""

from copy import copy
from pathlib import Path
from typing import Collection

import numpy as np
from bokeh import models
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import CheckboxGroup
from bokeh.palettes import Dark2_5
from bokeh.plotting import figure

from snat_sim import mock
from snat_sim.models import LightCurve, ReferenceCatalog, SNModel, StaticPWVTrans

# The color pallet to use when plotting
PALLETE = Dark2_5


class SimulationInputWidgets:
    """Collection of Bokeh Widgets for specifying input simulation parameters"""

    def __init__(self) -> None:
        """Instantiate bokeh widgets associated with the parent application element"""

        # User input widgets for setting model parameters
        self.z_slider = models.Slider(start=0.001, end=1, value=.55, step=.01, title='z')
        self.t0_slider = models.Slider(start=-10, end=10, value=-2, step=.01, title='t0')
        self.x0_slider = models.Slider(start=0.001, end=2, value=.25, step=.01, title='x0')
        self.x1_slider = models.Slider(start=-1, end=1, value=0.11, step=.01, title='x1')
        self.c_slider = models.Slider(start=-1, end=1, value=-.05, step=.01, title='c')
        self.pwv_slider = models.Slider(start=-0, end=100, value=7, step=.1, title='PWV')
        self.sampling_input = models.TextInput(value='4', title='Sampling (Days):')
        self.snr_input = models.TextInput(value='10.0', title='SNR:')
        self.sim_options_checkbox = models.CheckboxGroup(
            labels=["Plot SNR", 'Calibrate to Reference Catalog'], active=[0, 1])
        self.plot_model_button = models.Button(label='Plot Light-Curve', button_type='success')

    def as_column(self, width=320, height=1000, sizing_mode="fixed", **kwargs) -> column:
        """Return the application element as a column

        Args:
            All arguments are passed to the returned ``column`` instance
        """

        column_header = models.Div(text=(
            '<h2>Simulated Parameters</h2>'
            '<p>Use the controls below to vary the input parameters to the simulated light-curve. '))

        return column(
            column_header,
            self.z_slider,
            self.t0_slider,
            self.x0_slider,
            self.x1_slider,
            self.c_slider,
            self.pwv_slider,
            self.sampling_input,
            self.snr_input,
            self.sim_options_checkbox,
            self.plot_model_button,
            width=width,
            height=height,
            sizing_mode=sizing_mode,
            **kwargs
        )


class FittedParamWidgets:
    """Collection of Bokeh Widgets for input fitting parameters"""

    def __init__(self) -> None:
        """Instantiate bokeh widgets associated with the parent application element"""

        # Define widgets
        self.param_checkbox_group = CheckboxGroup(labels=['t0', 'x0', 'x1', 'c'], active=[0, 1, 2, 3], inline=True)
        self.t0_slider = models.Slider(start=-10, end=10, value=0, step=1E-4, title='t0')
        self.x0_slider = models.Slider(start=0.001, end=2, value=.1, step=1E-4, title='x0')
        self.x1_slider = models.Slider(start=-1, end=1, value=0, step=1E-4, title='x1')
        self.c_slider = models.Slider(start=-1, end=1, value=0, step=1E-4, title='c')
        self.pwv_slider = models.Slider(start=-0, end=100, value=4, step=.1, title='PWV')
        self.min_t0_input = models.TextInput(value=str(self.t0_slider.start), default_size=110)
        self.max_t0_input = models.TextInput(value=str(self.t0_slider.end), default_size=110)
        self.min_x0_input = models.TextInput(value=str(self.x0_slider.start), default_size=110)
        self.max_x0_input = models.TextInput(value=str(self.x0_slider.end), default_size=110)
        self.min_x1_input = models.TextInput(value=str(self.x1_slider.start), default_size=110)
        self.max_x1_input = models.TextInput(value=str(self.x1_slider.end), default_size=110)
        self.min_c_input = models.TextInput(value=str(self.c_slider.start), default_size=110)
        self.max_c_input = models.TextInput(value=str(self.c_slider.end), default_size=110)
        self.plot_model_button = models.Button(label='Plot Current Values', button_type='warning')
        self.run_fit_button = models.Button(label='Fit Light-Curve', button_type='success')

    def get_params_to_vary(self) -> list[str]:
        """Return a list of parameters the user has specified should be varied"""

        return self.param_checkbox_group.labels[self.param_checkbox_group.active]

    def get_params_boundaries(self):
        """Return a list of user specified parameter boundaries to be used when fitting"""

        return {
            't0': [self.min_t0_input, self.max_t0_input],
            'x0': [self.min_x0_input, self.max_x0_input],
            'x1': [self.min_x1_input, self.max_x1_input],
            'c': [self.min_c_input, self.max_c_input]
        }

    def as_column(self, width=320, height=1000, sizing_mode="fixed", **kwargs) -> column:
        """Return the application element as a column

        Args:
            All arguments are passed to the returned ``column`` instance
        """

        column_header = models.Div(text=(
            '<h2>Fitted Parameters</h2>'
            '<p>Current parameter values are used as an initial guess when fitting. '))

        return column(
            column_header,
            row(models.Div(text='Vary Params:'), self.param_checkbox_group),
            models.Div(text='Parameter Bounds:'),
            row(models.Div(text='t0:'), self.min_t0_input, self.max_t0_input),
            row(models.Div(text='x0:'), self.min_x0_input, self.max_x0_input),
            row(models.Div(text='x1:'), self.min_x1_input, self.max_x1_input),
            row(models.Div(text='c:'), self.min_c_input, self.max_c_input),
            models.Div(text='Parameter Values:'),
            self.t0_slider,
            self.x0_slider,
            self.x1_slider,
            self.c_slider,
            self.pwv_slider,
            self.plot_model_button,
            self.run_fit_button,
            width=width,
            height=height,
            sizing_mode=sizing_mode,
            **kwargs)


class ResultsPanel:
    """Collection of bokeh widgets for plotting simulation/fitting results"""

    def __init__(self) -> None:
        """Instantiate bokeh widgets associated with the parent application element"""

        # Track plotted simulation data
        self._plotted_lc_sims = []
        self._plotted_spec_sims = []

        # Track plotted fit data
        self._plotted_lc_fits = []
        self._plotted_spec_fits = []

        self.light_curve_figure = figure(
            plot_height=400,
            plot_width=700,
            sizing_mode='scale_both',
            toolbar_location='above')

        self.spectrum_figure = figure(
            plot_height=200,
            plot_width=700,
            sizing_mode='scale_both',
            toolbar_location='above')

        self.fit_results = models.Div()

    def clear_plotted_sim(self) -> None:
        """Remove simulated data from all plots"""

        while self._plotted_lc_sims:
            self.light_curve_figure.renderers.remove(self._plotted_lc_sims.pop())

        while self._plotted_spec_sims:
            self.spectrum_figure.renderers.remove(self._plotted_spec_sims.pop())

    def clear_plotted_fit(self) -> None:
        """Remove model fits from all plots"""

        while self._plotted_lc_fits:
            self.light_curve_figure.renderers.remove(self._plotted_lc_fits.pop())

        while self._plotted_spec_fits:
            self.spectrum_figure.renderers.remove(self._plotted_spec_fits.pop())

    def plot_simulated_lc(self, model: SNModel, light_curve: LightCurve, bands: Collection[str]) -> None:
        """Plot a simulated supernova light-curve

        Args:
            model: The supernova model to plot
            light_curve: The simulated light-curve
        """

        self.clear_plotted_sim()

        # Update the main plot with simulated flux data
        sim_as_astropy = light_curve.to_astropy()
        for band, color in zip(bands, PALLETE):
            band_data = sim_as_astropy[sim_as_astropy['band'] == band]
            x = band_data['time']
            y = band_data['flux']
            yerr = band_data['fluxerr']

            circ = self.light_curve_figure.circle(x=x, y=y, color=color, legend_label=band)
            err_bar = self.light_curve_figure.multi_line(
                np.transpose([x, x]).tolist(),
                np.transpose([y - yerr, y + yerr]).tolist(),
                color=color)

            self._plotted_lc_sims.append(circ)
            self._plotted_lc_sims.append(err_bar)

        # Update plot of simulated spectrum
        wave = np.arange(model.minwave(), model.maxwave())
        spec_line = self.spectrum_figure.line(x=wave, y=model.flux(0, wave), legend_label='Observed')
        self._plotted_spec_sims.append(spec_line)

    def plot_model_fit(self, model: SNModel, bands: Collection[str]) -> None:
        """Plot a fitted supernova model

        Args:
            model: The supernova model to plot
            bands: The photometric band passes to fit
        """

        self.clear_plotted_fit()

        # Plot photometric light-curves in each band
        time_arr = np.arange(-25, 55, .5)
        for band, color in zip(bands, PALLETE):
            line = self.light_curve_figure.line(
                x=time_arr,
                y=model.bandflux(band, time_arr, zp=30, zpsys='ab'),
                color=color
            )

            self._plotted_lc_fits.append(line)

        # Plot the spectrum
        wave = np.arange(model.minwave(), min(model.maxwave(), 12000))
        spec_line = self.spectrum_figure.line(x=wave, y=model.flux(0, wave), color='red', legend_label='Model')
        self._plotted_spec_fits.append(spec_line)

    def as_column(self, height=1200, sizing_mode="scale_width", **kwargs) -> column:
        """Return the application element as a column

        Args:
            All arguments are passed to the returned ``column`` instance
        """

        return column(
            self.light_curve_figure,
            self.spectrum_figure,
            self.fit_results,
            height=height,
            sizing_mode=sizing_mode,
            **kwargs)


class Application:
    """Core application logic for simulating and fitting supernovae"""

    sncosmo_source = 'salt2-extended'
    photometric_bands = tuple(f'lsst_hardware_{b}' for b in 'ugriz')
    reference_catalog = ReferenceCatalog('G2', 'M5', 'K2')
    header_path = Path(__file__).parent / "lc_fit_header.html"

    def __init__(self) -> None:
        """Instantiate the parent application and populate the given document"""

        self._simulated_lc = None
        self.sn_model = SNModel(
            self.sncosmo_source,
            effects=[StaticPWVTrans()],
            effect_frames=['obs'],
            effect_names=[''])

        self.sim_widgets = SimulationInputWidgets()
        self.fit_widgets = FittedParamWidgets()
        self.plots = ResultsPanel()

        # Instantiate application interface and behavior
        self._init_callbacks()
        self._init_layout()

    def _init_layout(self) -> None:
        """Populate the current document with the application GUI"""

        # Create layouts for different sections of the GUI
        header_div = models.Div(text=self.header_path.read_text(), sizing_mode="stretch_width")
        left_column = self.sim_widgets.as_column()
        center_column = self.plots.as_column()
        right_column = self.fit_widgets.as_column()

        # noinspection PyTypeChecker
        doc_layout = layout([
            [header_div],
            [left_column, center_column, right_column]
        ], sizing_mode="scale_both")

        curdoc().add_root(doc_layout)

    def _init_callbacks(self) -> None:
        """Initialize callbacks between application widgets"""

        self.sim_widgets.plot_model_button.on_click(self.simulate_lc_callback)
        self.fit_widgets.plot_model_button.on_click(self.plot_fit_callback)
        self.fit_widgets.run_fit_button.on_click(self.fit_lc_callback)

    def simulate_lc_callback(self, event=None):
        """Simulate and plot a new SN light-curve"""

        # Create a model using simulation parameters from the GUI
        model = copy(self.sn_model)
        model.set(
            z=self.sim_widgets.z_slider.value,
            t0=self.sim_widgets.t0_slider.value,
            x0=self.sim_widgets.x0_slider.value,
            x1=self.sim_widgets.x1_slider.value,
            c=self.sim_widgets.c_slider.value,
            pwv=self.sim_widgets.pwv_slider.value
        )

        # Simulate a light-curve
        snr = float(self.sim_widgets.snr_input.value)
        time_sampling = float(self.sim_widgets.sampling_input.value)
        cadence = mock.create_mock_cadence(np.arange(-10, 51, time_sampling), self.photometric_bands)
        self._simulated_lc = model.simulate_lc(cadence, fixed_snr=snr, scatter=False)

        # Scale flux by reference star
        if 1 in self.sim_widgets.sim_options_checkbox.active:
            self._simulated_lc = self.reference_catalog.calibrate_lc(self._simulated_lc, model['pwv'])

        # Update the main plot with simulated flux data
        self.plots.plot_simulated_lc(model, self._simulated_lc, self.photometric_bands)

        # Match fitted param sliders to sim param sliders
        self.fit_widgets.t0_slider.update(value=self.sim_widgets.t0_slider.value)
        self.fit_widgets.x0_slider.update(value=self.sim_widgets.x0_slider.value)
        self.fit_widgets.x1_slider.update(value=self.sim_widgets.x1_slider.value)
        self.fit_widgets.c_slider.update(value=self.sim_widgets.c_slider.value)
        self.fit_widgets.pwv_slider.update(value=self.sim_widgets.pwv_slider.value)

    def fit_lc_callback(self, event=None):
        """Fit and plot the SN model to the simulated light-curve"""

        if not self._simulated_lc:
            return

        model = copy(self.sn_model)
        model.set(
            z=self.sim_widgets.z_slider.value,
            t0=self.fit_widgets.t0_slider.value,
            x0=self.fit_widgets.x0_slider.value,
            x1=self.fit_widgets.x1_slider.value,
            c=self.fit_widgets.c_slider.value,
            pwv=self.fit_widgets.pwv_slider.value
        )

        # Simulate a light-curve
        fit_result = model.fit_lc(
            data=self._simulated_lc,
            vparam_names=self.fit_widgets.get_params_to_vary(),
            bounds=self.fit_widgets.get_params_boundaries()
        )

        # Update slider values and plot the result
        self.fit_widgets.t0_slider.value = fit_result.parameters['t0']
        self.fit_widgets.x0_slider.value = fit_result.parameters['x0']
        self.fit_widgets.x1_slider.value = fit_result.parameters['x1']
        self.fit_widgets.c_slider.value = fit_result.parameters['c']
        self.plot_fit_callback()

    def plot_fit_callback(self, event=None) -> None:
        """Plot the current fitted SN model"""

        model = copy(self.sn_model)
        model.set(
            z=self.sim_widgets.z_slider.value,
            t0=self.fit_widgets.t0_slider.value,
            x0=self.fit_widgets.x0_slider.value,
            x1=self.fit_widgets.x1_slider.value,
            c=self.fit_widgets.c_slider.value,
            pwv=self.fit_widgets.pwv_slider.value
        )

        self.plots.plot_model_fit(model, bands=self.photometric_bands)


Application()
