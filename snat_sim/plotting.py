"""The ``plotting`` module provides functions for generating plots of
analysis results that are visually consistent and easily reproducible.

Plotting Function Summaries
---------------------------

+------------------------------+----------------------------------------------+
| Function                     | Description                                  |
+==============================+==============================================+
| ``plot_delta_mag_vs_z``      | Single panel, multi-line plot of change in   |
|                              | magnitude vs z. Color coded by PWV.          |
+------------------------------+----------------------------------------------+
| ``plot_delta_mag_vs_pwv``    | Single panel, multi-line plot for change in  |
|                              | magnitude vs PWV. Color coded by redshift.   |
+------------------------------+----------------------------------------------+
| ``plot_derivative_mag_vs_z`` | Single panel, multi-line plot of slope in    |
|                              | delta magnitude vs z. Color coded by PWV.    |
+------------------------------+----------------------------------------------+
| ``plot_pwv_mag_effects``     | Multi panel plot with a column for each band |
|                              | and a row for each of the first three plots  |
|                              | in this table.                               |
+------------------------------+----------------------------------------------+
| ``plot_spectral_template``   | Plot the salt2-extended spectral template.   |
|                              | Overlay PWV and bandpass throughput curves.  |
+------------------------------+----------------------------------------------+
| ``plot_magnitude``           | Multi-panel plot showing with a column for   |
|                              | each band. Top row shows simulated magnitudes|
|                              | vs Redshift. Bottom row shows mag vs PWV.    |
+------------------------------+----------------------------------------------+
| ``plot_fitted_params``       | Multi-panel plot showing subplots for each   |
|                              | salt2 parameter vs redshift. Multiple lines  |
|                              | included for different PWV values.           |
+------------------------------+----------------------------------------------+
| ``plot_delta_colors``        | Shows the change in color as a function of   |
|                              | redshift color coded by PWV.                 |
+------------------------------+----------------------------------------------+
| ``plot_delta_x0``            | Plot the variation in x0 as a function of    |
|                              | redshift and PWV.                            |
+------------------------------+----------------------------------------------+
| ``plot_delta_mu``            | Plot the variation in fitted distance modulus|
|                              | as a function of redshift and PWV.           |
+------------------------------+----------------------------------------------+
| ``plot_delta_colors``        | Plot the change in fitted SN color           |
|                              | as a function of redshift and PWV.           |
+------------------------------+----------------------------------------------+
| ``plot_delta_colors``        | Plot PWV measurements taken over a single    |
|                              | year as a function of time.                  |
+------------------------------+----------------------------------------------+

Module Docs
-----------
"""

from datetime import datetime
from datetime import timedelta

import matplotlib.dates as mdates
import numpy as np
import sncosmo
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from pwv_kpno.defaults import v1_transmission
from pytz import utc

from . import constants as const, filters, lc_simulation

filters.register_lsst_filters(force=True)


def multi_line_plot(x_arr, y_arr, z_arr, axis, label=None):
    """Plot a 2d y array vs a 1d x array

    Lines are color coded according to values of a 2d z array

    Args:
        x_arr (ndarray): A 1d array
        y_arr (ndarray): A 2d array
        z_arr (ndarray): A 2d array
        axis     (Axis): Axis to plot on
        label     (str): Optional label to format with ``z`` value
    """

    # noinspection PyUnresolvedReferences
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_arr)))
    for z, y, color in zip(z_arr, y_arr, colors):
        if label:
            axis.plot(x_arr, y, label=label.format(z), c=color)

        else:
            axis.plot(x_arr, y, c=color)

    axis.set_xlim(x_arr[0], x_arr[-1])


def plot_delta_mag_vs_z(pwv_arr, z_arr, delta_mag_arr, axis=None, label=None):
    """Plot the change in apparent mag due to PWV vs redshift

    Args:
        pwv_arr       (ndarray): Array of PWV values
        z_arr         (ndarray): Array of redshift values
        delta_mag_arr (ndarray): Array of delta mag values
        axis             (Axis): Optionally plot on a given axis
        label             (str): Optional label to format with PWV
    """

    if axis is None:
        axis = plt.gca()

    multi_line_plot(z_arr, delta_mag_arr, pwv_arr, axis, label)
    axis.set_xlabel('Redshift', fontsize=20)
    axis.set_xlim(min(z_arr), max(z_arr))
    axis.set_ylabel(r'$\Delta m$', fontsize=20)


def plot_delta_mag_vs_pwv(pwv_arr, z_arr, delta_mag_arr, axis=None, label=None):
    """Plot the change in mag due to PWV as a function of pwv

    Args:
        pwv_arr       (ndarray): Array of PWV values
        z_arr         (ndarray): Array of redshift values
        delta_mag_arr (ndarray): Array of delta mag values
        axis             (Axis): Optionally plot on a given axis
        label             (str): Optional label to format with redshift
    """

    if axis is None:
        axis = plt.gca()

    multi_line_plot(pwv_arr, delta_mag_arr.T, z_arr, axis, label)
    axis.set_xlabel('PWV', fontsize=20)
    axis.set_xlim(min(pwv_arr), max(pwv_arr))
    axis.set_ylabel(r'$\Delta m$', fontsize=20)


# noinspection PyUnusedLocal
def plot_derivative_mag_vs_z(pwv_arr, z_arr, slope_arr, axis=None):
    """Plot the delta mag / delta PWV as a function of redshift

    Args:
        pwv_arr   (ndarray): Array of PWV values
        z_arr     (ndarray): Array of redshift values
        slope_arr (ndarray): Slope of delta mag at reference PWV
        axis         (Axis): Optionally plot on a given axis
    """

    if axis is None:
        axis = plt.gca()

    axis.plot(z_arr, slope_arr)
    axis.set_xlabel('Redshift', fontsize=20)
    axis.set_xlim(min(z_arr), max(z_arr))
    axis.set_ylabel(r'$\frac{\Delta \, m}{\Delta \, PWV} |_{PWV = 4 mm}$', fontsize=20)


def plot_pwv_mag_effects(pwv_arr, z_arr, delta_mag, slopes, bands, figsize=(10, 8)):
    """Plot the effects of PWV on SN magnitudes

    ``delta_mag`` is expected to have band names as keys, and 2d arrays as
    values. Each array should represent the change in magnitude for each
    given PWV and redshift

    Args:
        pwv_arr  (ndarray): PWV values used in the calculation
        z_arr    (ndarray): Redshift values used in the calculation
        delta_mag   (dict): Dictionary with delta mag for each band
        slopes   (ndarray): Slope in delta_mag for each redshift
        bands      (bands): Order of bands to plot
        figsize    (tuple): The size of the figure

    returns:
        Tuple with a matplotlib figure and axis
    """

    fig, axes = plt.subplots(3, len(delta_mag), figsize=figsize)
    top_reference_ax = axes[0, 0]
    middle_reference_ax = axes[1, 0]
    bottom_reference_ax = axes[2, 0]

    # Plot data
    for band, axes_column in zip(bands, axes.T):
        top_ax, middle_ax, bottom_ax = axes_column

        # First row
        plot_delta_mag_vs_z(pwv_arr, z_arr, delta_mag[band], top_ax, label='{:g} mm')
        top_ax.axhline(0, linestyle='--', color='k', label='4 mm')
        top_ax.set_title(f'{band[-1]}-band')
        top_ax.set_xlabel('Redshift', fontsize=12)
        top_ax.set_ylabel('')

        # Middle row
        plot_delta_mag_vs_pwv(pwv_arr, z_arr, delta_mag[band], middle_ax, label='z = {:g}')
        top_ax.axvline(4, linestyle='--', color='k')
        middle_ax.set_xlabel('PWV', fontsize=12)
        middle_ax.set_ylabel('')

        # Bottom row
        plot_derivative_mag_vs_z(pwv_arr, z_arr, slopes[band], bottom_ax)
        bottom_ax.set_xlabel('Redshift', fontsize=12)
        bottom_ax.set_ylabel('')

        # Share axes
        top_ax.get_shared_y_axes().join(top_ax, top_reference_ax)
        middle_ax.get_shared_y_axes().join(middle_ax, middle_reference_ax)
        bottom_ax.get_shared_y_axes().join(bottom_ax, bottom_reference_ax)

        top_ax.get_shared_x_axes().join(top_ax, top_reference_ax)
        bottom_ax.get_shared_x_axes().join(bottom_ax, top_reference_ax)

    top_reference_ax.autoscale()  # To reset y-range
    top_reference_ax.set_xlim(0.1, 1.1)

    # Remove unnecessary tick marks
    for axis in axes.T[1:].flatten():
        axis.set_yticklabels([])

    # Add legends
    top_ax.legend(bbox_to_anchor=(1, 1.1))
    handles, labels = middle_ax.get_legend_handles_labels()
    middle_ax.legend(handles[::5], labels[::5], bbox_to_anchor=(1, 1.1))

    # Add y labels
    top_reference_ax.set_ylabel(r'$\Delta m \, \left(PWV,\, z\right)$', fontsize=12)
    middle_reference_ax.set_ylabel(r'$\Delta m \, \left(z,\, PWV\right)$', fontsize=12)
    bottom_reference_ax.set_ylabel(r'$\frac{\Delta \, m}{\Delta \, PWV} |_{4 mm}$', fontsize=12)
    plt.tight_layout()

    return fig, axes


# https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """Return a string representation of number in scientific notation"""

    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))

    coeff = round(num / float(10 ** exponent), decimal_digits)
    if coeff == 1:
        return r"$10^{{{}}}$".format(exponent)

    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def plot_spectral_template(source, wave_arr, z_arr, pwv, phase=0, resolution=2, figsize=(6, 4)):
    """Plot the a spectral template at several redshifts overlaid with PWV

    Args:
        source (Source, str): sncosmo source to use as spectral template
        wave_arr   (ndarray): The observer frame wavelengths to plot flux for in Angstroms
        z_arr      (ndarray): The redshifts to plot the template at
        pwv          (float): The PWV to plot the transmission function for
        phase        (float): The phase of the template to plot
        resolution   (float): The resolution of the atmospheric model
        figsize      (tuple): The size of the figure

    Returns:
        Tuple with a matplotlib figure and axis
    """

    fig, (top_ax, bottom_ax) = plt.subplots(
        nrows=2, figsize=figsize, sharex='row',
        gridspec_kw={'height_ratios': [4, 1.75]})

    # Plot spectral template at given redshifts
    model = sncosmo.Model(source)
    flux_scale = 1e-13
    for i, z in enumerate(reversed(z_arr)):
        color = f'C{len(z_arr) - i - 1}'
        model.set(z=z)
        flux = model.flux(phase, wave_arr) / flux_scale
        top_ax.fill_between(wave_arr, flux, color=color, alpha=.8)
        top_ax.plot(wave_arr, flux, label=f'z = {z}', color=color, zorder=0)

    # Plot transmission function on twin axis at given wavelength resolution
    transmission = v1_transmission(pwv, res=resolution)
    twin_axis = top_ax.twinx()
    twin_axis.plot(transmission.index, transmission, alpha=0.75, color='grey')

    # Plot the band passes
    for b in 'rizy':
        band = sncosmo.get_bandpass(f'lsst_hardware_{b}')
        bottom_ax.plot(band.wave, band.trans, label=f'{b} Band')

    # Format top axis
    top_ax.set_ylim(0, 5)
    top_ax.set_xlim(min(wave_arr), max(wave_arr))
    top_ax.set_ylabel(f'Flux')
    top_ax.legend(loc='lower left', framealpha=1)

    # Format twin axis
    twin_axis.set_ylim(0, 1)
    twin_axis.set_ylabel('Transmission', rotation=-90, labelpad=12)
    plt.tight_layout()

    # Format bottom axis
    bottom_ax.set_ylim(0, 1)
    bottom_ax.set_xlabel(r'Wavelength $\AA$')
    bottom_ax.xaxis.set_minor_locator(MultipleLocator(500))
    bottom_ax.set_xticks(np.arange(4000, 11001, 2000))
    bottom_ax.legend(loc='lower left', framealpha=1)

    plt.subplots_adjust(hspace=0)

    return fig, np.array([top_ax, bottom_ax])


def plot_magnitude(mags, pwv, z, figsize=(9, 6)):
    """Plot simulated magnitudes vs Redshift and PWV

    Args:
        mags  (ndarray): Simulated magnitude values
        pwv   (ndarray): Array of PWV values
        z     (ndarray): Array of redshift values
        figsize (tuple): Size of the figure

    Returns:
        Tuple with a matplotlib figure and axis
    """

    fig, axes = plt.subplots(2, len(mags), figsize=figsize, sharey='row')
    for (band, mag_arr), (top_ax, bottom_ax) in zip(mags.items(), axes.T):
        top_ax.set_title(band)
        top_ax.set_xlabel('Redshift')
        multi_line_plot(z, mag_arr, pwv, top_ax, label='{:g} mm')

        bottom_ax.set_xlabel('PWV')
        multi_line_plot(pwv, mag_arr.T, z, bottom_ax, label='z = {:.2f}')

    axes[0][0].set_ylabel('Magnitude')
    axes[1][0].set_ylabel('Magnitude')

    # Add legends
    top_ax.legend(bbox_to_anchor=(1, 1.1))
    handles, labels = bottom_ax.get_legend_handles_labels()
    bottom_ax.legend(handles[::5], labels[::5], bbox_to_anchor=(1, 1.1))

    plt.tight_layout()
    return fig, axes


def plot_fitted_params(fitted_params, pwv_arr, z_arr, bands):
    """Plot fitted parameters as a function of Redshift.
    Color code by PWV.
    """

    # Parse the fitted parameters for easier plotting
    model = sncosmo.Model('salt2-extended')
    params_dict = {
        param: fitted_params[bands[0]][..., i] for
        i, param in enumerate(model.param_names)
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (param, param_vals) in zip(axes.flatten(), params_dict.items()):
        if param == 'x0':
            param_vals = -2.5 * np.log10(param_vals)
            param = r'-2.5 log$_{10}$(x$_{0}$)'

        multi_line_plot(z_arr, param_vals, pwv_arr, axis, label='z = {:g}')
        axis.set_xlabel('Redshift')
        axis.set_ylabel(param)

    correction_factor = const.betoule_alpha * params_dict['x1'] - const.betoule_beta * params_dict['c']
    multi_line_plot(z_arr, correction_factor, pwv_arr, axes[-1][-1], label='PWV = {:g} mm')

    label = f'{const.betoule_alpha} * $x_1$ - {const.betoule_beta} * $c$'
    axes[-1][-1].set_ylabel(label)
    axes[-1][-1].legend(bbox_to_anchor=(1, 1.1))

    plt.tight_layout()

    return fig, axes


def plot_delta_x0(source, pwv_arr, z_arr, params_dict):
    """Plot the variation in x0 as a function of redshift and PWV

    Args:
        source    (Source): Source corresponding to the provided parameters
        pwv_arr  (ndarray): Array of PWV values
        z_arr    (ndarray): Array of redshift values
        params_dict (dict): Dictionary with fitted parameters for each pwv and z
    """

    x0_cosmo = np.array([lc_simulation.calc_x0_for_z(z, source) for z in z_arr])
    delta_x0 = -2.5 * np.log10(params_dict['x0'] / x0_cosmo)

    fig, (left_ax, right_ax) = plt.subplots(ncols=2, sharey='col', figsize=(8, 4))
    multi_line_plot(z_arr, delta_x0, pwv_arr, left_ax, label='{} mm')
    multi_line_plot(pwv_arr, delta_x0.T, z_arr, right_ax, label='z = {:.2f}')

    left_ax.set_ylabel(r'-2.5 * $\log$($\frac{x_0}{x_{0,sim}}$)', fontsize=16)
    left_ax.set_xlabel('Redshift')
    right_ax.set_xlabel('PWV')

    handles, labels = left_ax.get_legend_handles_labels()
    labels = labels[::2]
    handles = handles[::2]
    left_ax.legend(handles, labels, bbox_to_anchor=(1, 1.1))

    handles, labels = right_ax.get_legend_handles_labels()
    labels = labels[::2]
    handles = handles[::2]
    right_ax.legend(handles, labels, bbox_to_anchor=(1, 1.1))

    plt.tight_layout()


def plot_delta_colors(pwv_arr, z_arr, mag_dict, colors, ref_pwv=0):
    """Plot the change in SN color as a function of redshift and PWV

    Args:
        pwv_arr    (ndarray): Array of PWV values
        z_arr      (ndarray): Array of redshift values
        mag_dict      (dict): Dictionary with magnitudes for each band
        colors (list[tuple]): Band combinations to plot colors for
        ref_pwv      (float): Plot values relative to given reference PWV
    """

    num_cols = len(colors)
    fig, axes = plt.subplots(ncols=num_cols, figsize=(4 * num_cols, 4), sharey='col')
    if num_cols == 1:
        axes = [axes]

    ref_idx = list(pwv_arr).index(ref_pwv)
    for axis, (band1, band2) in zip(axes, colors):
        color = mag_dict[band1] - mag_dict[band2]
        delta_color = color - color[ref_idx]
        multi_line_plot(z_arr, delta_color, pwv_arr, label='{} mm', axis=axis)

        axis.set_xlabel('Redshift', fontsize=14)
        axis.set_ylabel(fr'$\Delta$ ({band1[-1]} - {band2[-1]})', fontsize=14)

    axis.set_xlim(0, max(z_arr))
    axis.legend(bbox_to_anchor=(1, 1.1))
    plt.tight_layout()


# noinspection PyUnusedLocal
def plot_delta_mu(mu, pwv_arr, z_arr, cosmo=const.betoule_cosmo):
    """Plot the variation in fitted distance modulus as a function of redshift and PWV

    Args:
        mu      (ndarray): Array of distance moduli
        pwv_arr (ndarray): Array of PWV values
        z_arr   (ndarray): Array of redshift values
        cosmo (Cosmology): Astropy cosmology to compare results against
    """

    cosmo_mu = cosmo.distmod(z_arr).value
    delta_mu = mu - cosmo_mu

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
    mu_ax, delta_mu_ax, relative_mu_ax = axes

    multi_line_plot(z_arr, mu, pwv_arr, mu_ax)
    mu_ax.plot(z_arr, cosmo_mu, linestyle=':', color='k', label='Simulated')
    mu_ax.legend(framealpha=1)

    multi_line_plot(z_arr, delta_mu, pwv_arr, delta_mu_ax)
    delta_mu_ax.axhline(0, linestyle=':', color='k', label='Simulated')
    delta_mu_ax.legend(framealpha=1)

    multi_line_plot(z_arr, mu - mu[4], pwv_arr, relative_mu_ax, label='{:g} mm')
    relative_mu_ax.axhline(0, color='k', label=f'PWV={pwv_arr[4]}')
    relative_mu_ax.legend(framealpha=1, bbox_to_anchor=(1, 1.1))

    mu_ax.set_ylabel(r'$\mu$', fontsize=12)
    delta_mu_ax.set_ylabel(r'$\mu - \mu_{cosmo}$', fontsize=12)
    relative_mu_ax.set_ylabel(r'$\mu - \mu_{pwv_f}$', fontsize=12)
    for ax in axes:
        ax.set_xlabel('Redshift', fontsize=12)

    plt.tight_layout()


def plot_year_pwv_vs_time(pwv_series, figsize=(10, 4), missing=1):
    """Plot PWV measurements taken over a single year as a function of time

    Set ``missing=None`` to disable plotting of missing data windows.

    Args:
        pwv_series (Series): Measured PWV index by datetime
        figsize     (tuple): Size of the figure in inches
        missing     (float): Highlight time ranges larger than given number of
                             days with missing PWV

    Returns:
        Tuple with a matplotlib figure and axis
    """

    pwv_series = pwv_series.sort_index()

    # Calculate rolling average of PWV time series
    rolling_mean_pwv = pwv_series.rolling(window='7D').mean()

    # In practice these dates vary by year so the values used here are approximate
    year = pwv_series.index[0].year
    mar_equinox = datetime(year, 3, 20, tzinfo=utc)
    jun_equinox = datetime(year, 6, 21, tzinfo=utc)
    sep_equinox = datetime(year, 9, 22, tzinfo=utc)
    dec_equinox = datetime(year, 12, 21, tzinfo=utc)

    # Separate data based on season
    winter_pwv = pwv_series[(pwv_series.index < mar_equinox) | (pwv_series.index > dec_equinox)]
    spring_pwv = pwv_series[(pwv_series.index > mar_equinox) & (pwv_series.index < jun_equinox)]
    summer_pwv = pwv_series[(pwv_series.index > jun_equinox) & (pwv_series.index < sep_equinox)]
    fall_pwv = pwv_series[(pwv_series.index > sep_equinox) & (pwv_series.index < dec_equinox)]

    print(f'Winter Average: {winter_pwv.mean(): .2f} +\\- {winter_pwv.std() : .2f} mm')
    print(f'Spring Average: {spring_pwv.mean(): .2f} +\\- {spring_pwv.std() : .2f} mm')
    print(f'Summer Average: {summer_pwv.mean(): .2f} +\\- {summer_pwv.std() : .2f} mm')
    print(f'Fall Average:  {fall_pwv.mean(): .2f} +\\- {fall_pwv.std() : .2f} mm')

    fig, axis = plt.subplots(figsize=figsize)
    axis.set_ylabel('Median Folded PWV (mm)')
    axis.set_xlabel('Time of Year')
    axis.set_ylim(0, 20)
    ylow, yhigh = axis.get_ylim()

    # Plot windows of missing pwv
    if missing:
        missing_interval = timedelta(days=missing)
        year_start = datetime(year, 1, 1, tzinfo=utc)
        year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=utc)

        delta_t = pwv_series.index[1:] - pwv_series.index[:-1]
        start_indices = np.where(delta_t > missing_interval)[0]
        for index in start_indices:
            start_time = pwv_series.index[index]
            end_time = pwv_series.index[index + 1]
            axis.fill_between(
                x=[start_time, end_time], y1=[ylow, ylow], y2=[yhigh, yhigh],
                color='lightgrey', alpha=0.5, zorder=0)

        if (pwv_series.index[0] - year_start) > missing_interval:
            axis.fill_between(
                x=[year_start, pwv_series.index[0]],
                y1=[ylow, ylow], y2=[yhigh, yhigh],
                color='lightgrey', alpha=0.5, zorder=0)

        if (year_end - pwv_series.index[-1]) > missing_interval:
            axis.fill_between(
                x=[pwv_series.index[-1], year_end],
                y1=[ylow, ylow], y2=[yhigh, yhigh],
                color='lightgrey', alpha=0.5, zorder=0)

    # Plot measured PWV
    for equinox_date in (mar_equinox, jun_equinox, sep_equinox, dec_equinox):
        axis.axvline(equinox_date, linestyle='--', color='k', zorder=1)

    # Plot rolling average
    axis.scatter(pwv_series.index, pwv_series, s=1, alpha=.2, label='Median PWV', zorder=2)
    axis.plot(rolling_mean_pwv.index, rolling_mean_pwv, color='C1', label='Rolling Avg.', zorder=3, linewidth=2)

    # Plot seasonal average
    # winter is plotted separately because it spans the new year
    if not winter_pwv.empty:
        winter_avg = winter_pwv.mean()
        winter_std = winter_pwv.std()
        winter_subset = pwv_series[pwv_series.index < mar_equinox]
        winter_x = winter_subset.index.max() - (winter_subset.index.max() - winter_subset.index.min()) / 2
        axis.errorbar([winter_x], [winter_avg], yerr=[winter_std], color='k', zorder=4, linewidth=2, capsize=10,
                      capthick=2)
        axis.scatter([winter_x], [winter_avg], color='k', s=100, marker='+', zorder=4, label='Seasonal Avg.')

    for season in (spring_pwv, summer_pwv, fall_pwv):
        if season.empty:
            continue

        avg = season.mean()
        std = season.std()
        x = season.index.max() - (season.index.max() - season.index.min()) / 2
        axis.errorbar([x], [avg], yerr=[std], color='k', zorder=4, linewidth=2, capsize=10, capthick=2)
        axis.scatter([x], [avg], color='k', s=100, marker='+', zorder=4)

    # Format x labels to be three letter month abbreviations
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter('%b')
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.legend(framealpha=1)

    axis.twinx().set_ylim(axis.get_ylim())
    return fig, axis
