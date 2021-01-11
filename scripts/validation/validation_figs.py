from pathlib import Path

import numpy as np
import pandas as pd
from bokeh import models
from bokeh.layouts import gridplot
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

# Point at validation data directory
VALIDATION_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'validation'
TOOLTIPS = [("SNID", "@snid"), ("Chisq", "@chisq"), ("DOF", "@ndof")]
TOOLS = "pan,box_zoom,wheel_zoom,save,box_select,lasso_select,reset,help"


def corner_plot(data, x_vals=None, y_vals=None, x_labels=None, y_labels=None, figsize=300, **kwargs):
    """Interactive corner plot

    Defaults to plotting all columns of the given dataframe.

    Args:
        data: The data to plot
        x_vals: Column names to use as x values
        y_vals: Column names to use as y values
        x_labels: Optional axis labels to use for x values
        y_labels: Optional axis labels to use for y values
        figsize: Size of each sub-plotted figure
        **kwargs: And arguments for constructing bokeh circle plots

    Returns:
        A grid of bokeh figures
    """

    # Default to plotting all data in the given data frame
    x_vals = x_vals or data.columns
    y_vals = y_vals or data.columns

    # Default to using dataframe column names as axis labels
    x_labels = x_labels or x_vals
    y_labels = y_labels or y_vals

    source = ColumnDataSource(data=data)
    plot_layout = []

    # Iterate over rows from top to bottom
    for i, (y, y_label) in enumerate(zip(y_vals, y_labels)):
        new_row = []

        # Iterate over columns from left to right. Only iterate over subplots in the lower triangle.
        for j, (x, x_label) in enumerate(zip(x_vals[:i + 1], x_labels[:i + 1])):
            fig = figure(
                plot_width=figsize,
                plot_height=figsize,
                y_range=new_row[0].y_range if j else None,
                x_range=plot_layout[i - 1][j].x_range if i and (i > j) else None,
                tooltips=TOOLTIPS,
                tools=TOOLS
            )

            fig.circle(x, y, source=source, **kwargs)

            # Assign axis labels only to subplots along the figure boundary
            if i == len(x_vals) - 1:
                fig.xaxis.axis_label = x_label

            if j == 0:
                fig.yaxis.axis_label = y_label

            new_row.append(fig)

        plot_layout.append(new_row)

    return gridplot(plot_layout)


def show_validation_figure(path, params=['z', 'x0', 'x1', 'c']):
    pipeline_data = pd.read_csv(path).replace(-99.99, np.nan)
    sim_cols = [f'sim_{p}' for p in params]
    sim_labels = [f'Simulated {p}' for p in params]
    fit_cols = [f'fit_{p}' for p in params]
    fit_labels = [f'Fitted {p}' for p in params]

    header_div = models.Div(text=f'<h2>Pipeline Validation: {path.name}</h2>')
    figure = corner_plot(pipeline_data, sim_cols, fit_cols, sim_labels, fit_labels, size=5)
    doc_layout = layout([
        [header_div],
        [figure]
    ])

    show(doc_layout)


if __name__ == '__main__':
    show_validation_figure(VALIDATION_DATA_DIR / 'pwv_sim_0_fit_0.csv')
    show_validation_figure(VALIDATION_DATA_DIR / 'pwv_sim_4_fit_4.csv')
    show_validation_figure(VALIDATION_DATA_DIR / 'pwv_sim_epoch_fit_4.csv')
