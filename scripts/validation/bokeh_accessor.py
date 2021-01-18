from typing import *

import numpy as np
import pandas as pd
from arviz.plots import kdeplot
from bokeh import layouts, models, plotting

StrColl = TypeVar('StrColl', str, Collection[str])
default_tooltips = [('SNID', '@snid'), ('Chisq', '@chisq'), ('DOF', '@ndof')]
default_tools = 'pan,box_zoom,wheel_zoom,save,box_select,lasso_select,reset,help'


class BasePlotter:
    """Base class for building plotting accessors"""

    def __init__(self, source: models.ColumnDataSource) -> None:
        self.source = source

    @staticmethod
    def make_figure(
            output_backend='webgl', tooltips=default_tooltips, tools=default_tools, **kwargs
    ) -> plotting.Figure:
        """Wrapper for creating bokeh figures that enforces customized default arguments"""

        return plotting.Figure(output_backend=output_backend, tooltips=tooltips, tools=tools, **kwargs)


class CornerPlotBuilder(BasePlotter):
    """Handles the construction of corner style plots"""

    def corner(
            self,
            x_vals: List[str],
            y_vals: List[str],
            x_labels: List[str] = None,
            y_labels: List[str] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            link_axes: bool = True,
            **kwargs
    ) -> models.GridBox:
        """Create an interactive corner plot

        Args:
            x_vals: Column names to use as x values
            y_vals: Column names to use as y values
            x_labels: Optional axis labels to use for x values
            y_labels: Optional axis labels to use for y values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to the same value as ``plot_width``)
            link_axes: Whether to link together subplot axes
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A grid of bokeh figures
        """

        # Default to a square aspect ratio
        plot_height = plot_height or plot_width
        subplot_width = plot_width // len(x_vals)
        subplot_height = plot_height // len(x_vals)

        plot_layout = []
        for row, (y, y_label) in enumerate(zip(y_vals, y_labels)):  # Iterate over rows from top to bottom
            new_row = self._make_corner_row(y, x_vals[:row + 1], subplot_width, subplot_height, **kwargs)
            plot_layout.append(new_row)

        self._assign_corner_labels(plot_layout, x_labels, y_labels)
        if link_axes:
            self._link_corner_axes(plot_layout)

        return layouts.gridplot(plot_layout)

    def _make_corner_row(
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
            fig = self.make_figure(plot_width=subplot_width, plot_height=subplot_height)
            fig.circle(x, y, source=self.source, **kwargs)
            new_row.append(fig)

        # noinspection PyTupleAssignmentBalance
        no_nan = ~np.isnan(self.source.data[x])
        m, b = np.ma.polyfit(self.source.data[x][no_nan], self.source.data[y][no_nan], 1)

        fig.add_layout(models.Slope(gradient=m, y_intercept=b, line_color='red'))
        fig.add_layout(models.Slope(gradient=1, y_intercept=0, line_color='grey', line_dash='dashed'))
        fig.title.text = f'y = {m:.2f}x + {b:.3f}'

        # Enforce a square aspect ratio for the sub-figure at the end of the row
        fig.y_range.start = fig.x_range.start
        fig.y_range.end = fig.x_range.end
        return new_row

    @staticmethod
    def _assign_corner_labels(
            plot_layout: List[List[plotting.Figure]], x_labels: List[str], y_labels: List[str]
    ) -> None:
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
    def _link_corner_axes(plot_layout: List[List[plotting.Figure]]) -> None:
        """Link the range of x/y axes sharing a column/row:

        Args:
            plot_layout: A 2-d list of bokeh figures to link the axes for
        """

        for i, row in enumerate(plot_layout):
            for j, figure in enumerate(row):
                if j:
                    figure.y_range = row[0].y_range

                if i and (i > j):
                    figure.x_range = plot_layout[i - 1][j].x_range


class ScatterPlotBuilder(BasePlotter):
    """Handles the construction of scatter plots"""

    def scatter(
            self,
            x_vals: StrColl,
            y_vals: StrColl,
            x_labels: Optional[StrColl] = None,
            y_labels: Optional[StrColl] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            link_axes: bool = True,
            contour: bool = False,
            **kwargs
    ) -> Union[models.GridBox, plotting.Figure]:
        """Create an interactive row of scatter plots

        Args:
            x_vals: Column names to use as x values
            y_vals: Column names to use as y values
            x_labels: Optional axis labels to use for x values
            y_labels: Optional axis labels to use for y values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to the same value as ``plot_width``)
            link_axes: Whether to link together subplot axes
            contour: Whether to add contours to the scatter plot
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A bokeh figure or a layout of figures
        """

        x_vals = np.atleast_1d(x_vals)
        y_vals = np.atleast_1d(y_vals)
        x_labels = np.atleast_1d(x_labels) if x_labels is not None else x_vals
        y_labels = np.atleast_1d(y_labels) if y_labels is not None else y_vals

        # Default to a square aspect ratio for each subplot
        width = int(plot_width / len(x_vals))
        height = plot_height or width

        arg_iter = zip(x_vals, y_vals, x_labels, y_labels)
        if contour:
            figs = [self._make_contour(*args, width, height, **kwargs) for args in arg_iter]

        else:
            figs = [self._make_scatter(*args, width, height, **kwargs) for args in arg_iter]

        if len(figs) == 1:
            return figs[0]

        if link_axes:
            self._link_scatter_axes(figs)

        return layouts.gridplot([figs])

    def _make_scatter(
            self, x: str, y: str, x_label: str, y_label: str,
            plot_width: int = 1200, plot_height: int = 1200,
            fig: Optional[plotting.Figure] = None, **kwargs
    ) -> plotting.Figure:
        """Make a scatter plot without any contours

        Args:
            x: Name of the value to plot along the x-axis
            y: Name of the value to plot along the y-axis
            x_label: Axis label for the x-axis
            y_label: Axis label for the y-axis
            plot_width: Width of the returned figure
            plot_height: Height of the returned figure
            fig: Optionally plot using an existing figure
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A bokeh Figure
        """

        fig = fig or self.make_figure(plot_width=plot_width, plot_height=plot_height)
        fig.circle(x, y, source=self.source, **kwargs)
        fig.xaxis.axis_label = x_label or x
        fig.yaxis.axis_label = y_label
        return fig

    def _make_contour(
            self, x: str, y: str, x_label: str, y_label: str, plot_width: int, plot_height: int, **kwargs
    ) -> plotting.Figure:
        """Create a Bokeh scatter plot with auto generated contours

        Args:
            x: Name of the value to plot along the x-axis
            y: Name of the value to plot along the y-axis
            x_label: Axis label for the x-axis
            y_label: Axis label for the y-axis
            plot_width: Width of the returned figure
            plot_height: Height of the returned
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A bokeh Figure
        """

        fig = self.make_figure(plot_width=plot_width, plot_height=plot_height)

        unique = pd.DataFrame(self.source.data)[[x, y]].drop_duplicates()
        _, glyphs = kdeplot.plot_kde(unique[x], unique[y], ax=fig, backend='bokeh', show=False, return_glyph=True)
        self._make_scatter(x, y, x_label, y_label, fig=fig, **kwargs)

        return fig

    @staticmethod
    def _link_scatter_axes(figures: List[plotting.Figure]) -> None:
        """Link the x and y axis ranges together for all plots

        Args:
            figures: list of bokeh figures
        """

        for fig in figures:
            fig.x_range = figures[0].x_range
            fig.y_range = figures[0].y_range


class HistogramBuilder(BasePlotter):
    """Handles the construction of histogram plots"""

    def histogram(
            self,
            x_vals: StrColl,
            x_labels: Optional[StrColl] = None,
            plot_width: int = 1200,
            plot_height: Optional[int] = None,
            link_axes: bool = True,
            bins: Optional[np.ndarray] = None,
            **kwargs
    ) -> Union[models.GridBox, plotting.Figure]:
        """Create an interactive row of histogram plots

        Args:
            x_vals: Column names to use as x values
            x_labels: Optional axis labels to use for x values
            plot_width: Width of the overall figure
            plot_height: Height of the overall figure (Defaults to the same value as ``plot_width``)
            link_axes: Whether to link together subplot axes
            bins: Bins to use for each histogram
            **kwargs: Any arguments for constructing bokeh circle plots

        Returns:
            A bokeh figure or a layout of figures
        """

        x_vals = np.atleast_1d(x_vals)
        x_labels = np.atleast_1d(x_labels) or x_vals
        bins = np.atleast_2d(bins)

        # Default to a square aspect ratio for each subplot
        width = int(plot_width / len(x_vals))
        height = plot_height or width

        arg_iter = zip(x_vals, x_labels, bins)
        figs = [self._make_histogram(x, xl, b, width, height, **kwargs) for x, xl, b in arg_iter]

        if len(figs) == 1:
            return figs[0]

        if link_axes:
            self._link_hist_axes(figs)

        return layouts.gridplot([figs])

    def _make_histogram(
            self, x: str, x_label: str, bins: Optional[np.array], plot_width: int, plot_height: int, **kwargs
    ) -> plotting.Figure:
        """Create a Bokeh scatter plot with auto generated contours

        Args:
            x: Name of the value to plot along the x-axis
            x_label: Optional axis label for the x-axis
            y_label: Optional axis label for the y-axis
            plot_width: Width of the returned figure
            plot_height: Height of the returned
            bins: Bins to use when constructing histogram sums
            **kwargs: Any arguments for constructing bokeh quad plots

        Returns:
            A bokeh Figure
        """

        fig = self.make_figure(plot_width=plot_width, plot_height=plot_height or plot_width, tooltips=None)
        hhist, hedges = np.histogram(self.source.data[x], bins=bins)
        fig.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, **kwargs)
        fig.xaxis.axis_label = x_label or x
        return fig

    @staticmethod
    def _link_hist_axes(figures: List[plotting.Figure]) -> None:
        """Link x axis ranges together for all plots

        Args:
            figures: list of bokeh figures
        """

        for fig in figures:
            fig.x_range = figures[0].x_range


@pd.api.extensions.register_dataframe_accessor('snat_sim_bokeh')
class Plotting(CornerPlotBuilder, ScatterPlotBuilder, HistogramBuilder):

    def __init__(self, pandas_obj: pd.DataFrame):
        super().__init__(models.ColumnDataSource(pandas_obj))
