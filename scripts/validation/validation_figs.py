from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from bokeh import layouts, models, plotting
from pwv_kpno.defaults import ctio

from snat_sim.models import PWVModel

# Point at validation data directory
CornerPlotFigures = List[List[plotting.Figure]]
VALIDATION_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'validation'
DEFAULT_PWV_MODEL = PWVModel.from_suominet_receiver(ctio, 2016, [2017])
default_tooltips = [("SNID", "@snid"), ("Chisq", "@chisq"), ("DOF", "@ndof")]
default_tools = "pan,box_zoom,wheel_zoom,save,box_select,lasso_select,reset,help"


def make_figure(**kwargs) -> plotting.Figure:
    """Enforces standard default arguments for new bokeh figures"""

    kwargs.setdefault('output_backend', 'webgl')
    kwargs.setdefault('tooltips', default_tooltips)
    kwargs.setdefault('tools', default_tools)
    return plotting.Figure(**kwargs)


class CornerPlotBuilder:
    """Handles the construction of corner style plots using Bokeh"""

    def corner_plot(
            self,
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
            x_vals: Column names to use as x values
            y_vals: Column names to use as y values
            x_labels: Optional axis labels to use for x values
            y_labels: Optional axis labels to use for y values
            plot_width: Width of the overall make_figure
            plot_height: Height of the overall make_figure (Defaults to the same value as plot_width)
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
            plot_layout.append(self._create_new_row(y, x_vals[:row + 1], subplot_width, subplot_height, **kwargs))

        self._assign_subplot_labels(plot_layout, x_labels, y_labels)
        self._link_axes(plot_layout)
        return layouts.gridplot(plot_layout)

    def _create_new_row(
            self, y: str, x_vals: List[str], subplot_width: int, subplot_height: int, **kwargs
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
            fig.circle(x, y, source=self.source, **kwargs)
            new_row.append(fig)

        # Enforce a square aspect ratio for the subfigure at the end of the row
        new_row[-1].y_range.start = new_row[-1].x_range.start
        new_row[-1].y_range.end = new_row[-1].x_range.end
        return new_row

    @staticmethod
    def _assign_subplot_labels(plot_layout: CornerPlotFigures, x_labels: List[str], y_labels: List[str]) -> None:
        """Assign x and y axis labels to subplots along the make_figure boundary

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
    def _link_axes(plot_layout: CornerPlotFigures) -> None:
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


class ResidualsPlotBuilder:
    """Handles the construction of a scatter plot with associated residuals"""

    def residuals_plot(
            self,
            x: str,
            measured: str,
            modeled: str,
            residuals: str,
            y_label: str = None,
            x_label: str = None,
            plot_width: int = 1200,
            plot_height: int = 600
    ) -> models.Column:
        """Create an interactive scatter plot

        Args:
            x: Column name to use as x values
            measured: Column name to use for measured y values
            modeled: Column name to use for modeled y values
            residuals: Column name for data residuals
            x_label: Optional axis label to use for x values
            y_label: Optional axis label to use for y values
            plot_width: Width of the overall make_figure
            plot_height: Height of the overall make_figure

        Returns:
            A grid of bokeh figures
        """

        top_fig = self._build_top_subplot(x, measured, modeled, plot_width, plot_height)
        bottom_fig = self._build_bottom_subplot(x, residuals, plot_width, plot_height)
        bottom_fig.x_range = top_fig.x_range

        top_fig.yaxis.axis_label = y_label or measured
        bottom_fig.yaxis.axis_label = 'Residuals'
        bottom_fig.xaxis.axis_label = x_label or x
        return layouts.gridplot([[top_fig], [bottom_fig]])

    def _build_top_subplot(self, x, measured, modeled, plot_width, plot_height):
        top_fig = make_figure(plot_width=plot_width, plot_height=int(plot_height * .7))
        top_fig.circle(x, measured, source=self.source, legend_label='Measured')
        top_fig.circle(x, modeled, source=self.source, color='orange', legend_label='Modeled')
        top_fig.legend.click_policy = 'hide'
        return top_fig

    def _build_bottom_subplot(self, x, resid, plot_width, plot_height):
        bottom_fig = plotting.figure(plot_width=plot_width, plot_height=int(plot_height * .3))
        bottom_fig.circle(x, resid, source=self.source)
        return bottom_fig


class Validation(CornerPlotBuilder, ResidualsPlotBuilder):
    """Handles the construction of plots for pipeline output data"""

    def __init__(self, path):
        self.fit_results = pd.read_csv(path, index_col=0).replace(-99.99, np.nan)
        self.light_curves = self._load_light_curve_sims(VALIDATION_DATA_DIR / path.stem)

        # Combine all of the pipeline data into a single Bokeh data source
        combined_data = self.light_curves.join(self.fit_results)
        is_first_entry_for_sn = ~combined_data.index.duplicated()

        self.source = models.ColumnDataSource(data=combined_data)
        self.unique_view = models.CDSView(source=self.source, filters=[models.BooleanFilter(is_first_entry_for_sn)])

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

    def build_validation_page(self, params=('z', 'x0', 'x1', 'c')):
        parameter_validation = self.corner_plot(
            view=self.unique_view,
            x_vals=[f'sim_{p}' for p in params],
            y_vals=[f'fit_{p}' for p in params],
            x_labels=[f'Simulated {p}' for p in params],
            y_labels=[f'Fitted {p}' for p in params],
            size=5
        )

        doc_layout = layouts.layout([
            [parameter_validation]
        ])

        plotting.show(doc_layout)


if __name__ == '__main__':
    Validation(VALIDATION_DATA_DIR / 'pwv_sim_epoch_fit_4.csv').build_validation_page()
