from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from arviz.plots import kdeplot
from astropy.table import Table, vstack
from bokeh import layouts, models, plotting
from matplotlib import pyplot as plt
from pwv_kpno.defaults import ctio

from snat_sim.models import PWVModel

GridLayoutType = List[List[plotting.Figure]]


def make_figure(**kwargs) -> plotting.Figure:
    """Enforces standard default arguments for new bokeh figures"""

    default_tooltips = [('SNID', '@snid'), ('Chisq', '@chisq'), ('DOF', '@ndof')]
    default_tools = 'pan,box_zoom,wheel_zoom,save,box_select,lasso_select,reset,help'

    kwargs.setdefault('output_backend', 'webgl')
    kwargs.setdefault('tooltips', default_tooltips)
    kwargs.setdefault('tools', default_tools)
    return plotting.Figure(**kwargs)


class CornerPlotBuilder:
    """Handles the construction of corner style plots"""

    def __new__(
            cls,
            source,
            x_vals: List[str],
            y_vals: List[str],
            x_labels: List[str] = None,
            y_labels: List[str] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            **kwargs
    ) -> models.Column:
        """Create an interactive corner plot

        Args:
            source: Column data source to draw data from
            x_vals: Column names to use as x values
            y_vals: Column names to use as y values
            x_labels: Optional axis labels to use for x values
            y_labels: Optional axis labels to use for y values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to the same value as plot_width)
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A grid of bokeh figures
        """

        # Default to using dataframe column names as axis labels
        x_labels = x_labels or x_vals
        y_labels = y_labels or y_vals

        # Default to a square aspect ratio
        plot_height = plot_height or plot_width
        subplot_width = plot_width // len(x_vals)
        subplot_height = plot_height // len(x_vals)

        plot_layout = []
        for row, (y, y_label) in enumerate(zip(y_vals, y_labels)):  # Iterate over rows from top to bottom
            new_row = cls._create_new_row(source, y, x_vals[:row + 1], subplot_width, subplot_height, **kwargs)
            plot_layout.append(new_row)

        cls._assign_subplot_labels(plot_layout, x_labels, y_labels)
        cls._link_axes(plot_layout)
        return layouts.gridplot(plot_layout)

    @staticmethod
    def _create_new_row(
            source, y: str, x_vals: List[str], subplot_width: int, subplot_height: int, **kwargs
    ) -> List[plotting.Figure]:
        """Create a new row for the corner plot

        Args:
            y: Name of the data to plot along the y-axis of all figures in the row
            x_vals: List of x value names to use for each subplot in the row
            subplot_width: The width of each subplot
            subplot_height: The height of each subplot
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A list of bokeh figures with the same length as ``x_vals``
        """

        new_row = []
        for col, x in enumerate(x_vals):  # Iterate over columns from left to right
            fig = make_figure(plot_width=subplot_width, plot_height=subplot_height)
            fig.circle(x, y, source=source, **kwargs)
            new_row.append(fig)

        # noinspection PyTupleAssignmentBalance
        no_nan = ~np.isnan(source.data[x])
        m, b = np.ma.polyfit(source.data[x][no_nan], source.data[y][no_nan], 1)

        fig.add_layout(models.Slope(gradient=m, y_intercept=b, line_color='red'))
        fig.add_layout(models.Slope(gradient=1, y_intercept=0, line_color='grey', line_dash='dashed'))
        fig.title.text = f'y = {m:.2f}x + {b:.3f}'

        # Enforce a square aspect ratio for the sub-figure at the end of the row
        fig.y_range.start = fig.x_range.start
        fig.y_range.end = fig.x_range.end
        return new_row

    @staticmethod
    def _assign_subplot_labels(plot_layout: GridLayoutType, x_labels: List[str], y_labels: List[str]) -> None:
        """Assign x and y axis labels to subplots along the figure boundary

        Args:
            plot_layout: A 2-d list of bokeh figures in a corner plot (i.e., lower left triangular) layout
            x_labels: Labels to apply along the x-axis
            y_labels: Labels to apply along the y-axis
        """

        for i, ylabel in enumerate(y_labels):
            plot_layout[i][0].yaxis.axis_label = ylabel

        for j, xlabel in enumerate(x_labels):
            plot_layout[-1][j].xaxis.axis_label = xlabel

    @staticmethod
    def _link_axes(plot_layout: GridLayoutType) -> None:
        """Link the range of the x/y axes subplots sharing a column/row:

        Args:
            plot_layout: A 2-d list of bokeh figures in a corner plot (i.e., lower left triangular) layout
        """

        for i, row in enumerate(plot_layout):
            for j, figure in enumerate(row):
                if j:
                    figure.y_range = row[0].y_range

                if i and (i > j):
                    figure.x_range = plot_layout[i - 1][j].x_range


class ContourPlotBuilder:
    """Handles the construction of a row of scatter plots with KDE contours"""

    def __new__(
            cls,
            source,
            x_vals: List[str],
            y_vals: List[str],
            x_labels: List[str] = None,
            y_labels: List[str] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            **kwargs
    ) -> models.Column:
        """Create an interactive row of contoured scatter plots

        Args:
            source: Column data source to draw data from
            x_vals: Column names to use as x values
            y_vals: Column names to use as y values
            x_labels: Optional axis labels to use for x values
            y_labels: Optional axis labels to use for y values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to a square aspect ratio for each subplot)
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A grid of bokeh figures
        """

        # Default to a square aspect ratio for each subplot
        width = int(plot_width / len(x_vals))
        height = plot_height or width

        x_labels = x_labels or x_vals
        y_labels = y_labels or y_vals

        arg_iter = zip(x_vals, y_vals, x_labels, y_labels)
        figs = [cls._make_subplot(source, *args, width, height, **kwargs) for args in arg_iter]
        plotting_layout = [figs]

        cls._link_axes(plotting_layout)
        return layouts.gridplot(plotting_layout)

    @classmethod
    def _make_subplot(cls, source, x, y, x_label, y_label, plot_width, plot_height, **kwargs):
        """Create a Bokeh scatter plot of the given plot_height

        Args:
            source: Column data source to draw data from
            x: Name of the value to plot along the x-axis
            y: Name of the value to plot along the y-axis
            x_label: Optional axis label for the x-axis
            y_label: Optional axis label for the y-axis
            plot_width: Width of the returned figure
            plot_height: Height of the returned
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A grid of bokeh figures
        """

        fig = make_figure(plot_width=plot_width, plot_height=plot_height)

        unique = pd.DataFrame(source.data)[[x, y]].drop_duplicates()
        _, glyphs = kdeplot.plot_kde(unique[x], unique[y], ax=fig, backend='bokeh', show=False, return_glyph=True)

        fig.circle(x, y, source=source, **kwargs)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label
        return fig

    @staticmethod
    def _link_axes(plot_layout: GridLayoutType) -> None:
        """Link the range of the x/y axes subplots sharing a column/row:

        Args:
            plot_layout: list of bokeh figures in a corner plot (i.e., lower left triangular) layout
        """

        for row in plot_layout:
            for figure in row:
                figure.x_range = plot_layout[0][0].x_range
                figure.y_range = plot_layout[0][0].y_range


class ScatterPlotBuilder:

    def __new__(
            cls,
            source,
            x_vals: str,
            y_vals: Union[str, List[str]],
            x_label: str,
            y_label: str,
            legend_labels: List[str] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            **kwargs
    ) -> plotting.Figure:
        """Create an interactive row of contoured scatter plots

        Args:
            source: Column data source to draw data from
            x: Column names to use as x values
            y_vals: Column names to use as y values
            x_label: Axis label to use for x values
            y_label: Axis label to use for y values
            legend_labels: Optional custom legend labels
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to a square aspect ratio for each subplot)
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A grid of bokeh figures
        """

        height = plot_height or plot_width
        y_vals = np.atleast_1d(y_vals)
        legend_labels = legend_labels or y_vals

        default_color = kwargs.pop('color', None)
        colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        fig = make_figure(plot_width=plot_width, plot_height=height)
        for y, label in zip(y_vals, legend_labels):
            fig.circle(x_vals, y, source=source, legend_label=label, color=default_color or next(colors), **kwargs)

        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label
        fig.legend.click_policy = 'hide'
        return fig


class ResidualsPlotBuilder:
    """Handles the construction of a scatter plot with associated residuals"""

    def __new__(
            cls,
            source,
            x: str = 'time',
            sim_pwv: str = 'pwv',
            mod_pwv: str = 'pwv_model',
            residuals: str = 'pwv_resid',
            airmass: str = 'airmass',
            y_label: str = 'PWV',
            x_label: str = 'Time',
            plot_width: int = 1200,
            plot_height: int = 800
    ) -> models.Column:
        """Create an interactive scatter plot

        Args:
            x: Column name to use as x values
            sim_pwv: Column name to use for measured y values
            mod_pwv: Column name to use for modeled y values
            residuals: Column name for data residuals
            x_label: Optional axis label to use for x values
            y_label: Optional axis label to use for y values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure

        Returns:
            A grid of bokeh figures
        """

        top_fig = cls._build_top_subplot(source, x, sim_pwv, mod_pwv, plot_width, int(plot_height * .6))
        bottom_fig = cls._build_bottom_subplot(source, x, residuals, airmass, plot_width, int(plot_height * .4))
        bottom_fig.x_range = top_fig.x_range

        top_fig.yaxis.axis_label = y_label or sim_pwv
        bottom_fig.yaxis.axis_label = 'Residuals'
        bottom_fig.xaxis.axis_label = x_label or x
        return layouts.gridplot([[top_fig], [bottom_fig]])

    @staticmethod
    def _build_top_subplot(source, x, sim_pwv, mod_pwv, plot_width, plot_height):
        top_fig = make_figure(plot_width=plot_width, plot_height=plot_height)
        top_fig.circle(x, mod_pwv, source=source, legend_label='PWV Model (Zenith)')
        top_fig.circle(x, sim_pwv, source=source, color='orange', legend_label='Realized PWV')
        top_fig.legend.click_policy = 'hide'
        return top_fig

    @staticmethod
    def _build_bottom_subplot(source, x, resid, airmass, plot_width, plot_height):
        bottom_fig = make_figure(plot_width=plot_width, plot_height=plot_height)
        bottom_fig.circle(x, resid, source=source)
        bottom_fig.line(x, airmass, source=source, color='grey')
        return bottom_fig


class Validation:
    """Handles the construction of plots for pipeline output data"""

    def __init__(self, path, pwv_model):
        # Combine all of the pipeline data into a single Bokeh data source
        fit_results = pd.read_csv(path, index_col=0).replace(-99.99, np.nan)
        light_curves = self._load_light_curve_sims(validation_dir / path.stem)
        combined_data = light_curves.join(fit_results)

        combined_data['pwv_model'] = pwv_model.pwv_zenith(combined_data['time'])
        combined_data = combined_data.sort_values('time')

        self.source = models.ColumnDataSource(data=combined_data)

        # Create a view for selecting non-duplicate fit results
        is_unique_fit = ~combined_data.index.duplicated()
        self.fit_results_view = models.CDSView(source=self.source, filters=[models.BooleanFilter(is_unique_fit)])

    @staticmethod
    def _load_light_curve_sims(directory):
        light_curves = []
        for path in directory.glob('*.ecsv'):
            data = Table.read(path)
            data['snid'] = path.stem
            light_curves.append(data)

        all_data = vstack(light_curves, metadata_conflicts='silent').to_pandas(index='snid')
        all_data.index = all_data.index.astype('int64')
        return all_data

    def build_validation_page(self, params=('z', 'x0', 'x1', 'c'), contours=(['sim_c', 'fit_c'], ['sim_x1', 'fit_x1'])):
        param_scatter = CornerPlotBuilder(
            self.source,
            x_vals=[f'sim_{p}' for p in params],
            y_vals=[f'fit_{p}' for p in params],
            x_labels=[f'Simulated {p}' for p in params],
            y_labels=[f'Fitted {p}' for p in params],
            view=self.fit_results_view,
            size=5
        )

        param_contour = ContourPlotBuilder(
            self.source,
            x_vals=contours[0],
            y_vals=contours[1],
            view=self.fit_results_view)

        pwv_scatter = ScatterPlotBuilder(
            self.source, 'time', ['pwv_model', 'pwv'], 'Time', 'PWV', plot_height=500, alpha=.05)

        airmass_scatter = ScatterPlotBuilder(
            self.source, 'time', 'airmass', 'Time', 'Airmass', plot_height=300, color='grey', alpha=.1)

        doc_layout = layouts.layout([
            [param_scatter],
            [param_contour],
            [pwv_scatter],
            [airmass_scatter]
        ])

        plotting.show(doc_layout)


if __name__ == '__main__':
    validation_dir = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'validation'
    base_pwv_model = PWVModel.from_suominet_receiver(ctio, 2016, [2017])
    Validation(validation_dir / 'pwv_sim_epoch_fit_epoch.csv', base_pwv_model).build_validation_page()
