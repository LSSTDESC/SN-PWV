# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module provides general utilities for plotting data and registering
``sncosmo`` filters.
"""

from pathlib import Path

import numpy as np
import sncosmo
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

filter_dir = Path('.').resolve().parent / 'data/filters'


def register_decam_filters(force=False):
    """Register DECam filter profiles, CCD response, and fiducial ATM with sncosmo

    Args:
        force: Re-register a band if it is already registered
    """

    # Register each filter
    for filter_name in 'ugrizY':
        # Iterate over bands with and without the atmosphere
        for extension in ('', '_filter'):
            band_name = filter_name + extension
            filter_path = filter_dir / f'CTIO_DECam.{band_name}.dat'

            wave, transmission = np.genfromtxt(filter_path).T
            new_band = sncosmo.Bandpass(wave, transmission)
            new_band.name = 'DECam_' + band_name
            sncosmo.register(new_band, force=force)

    # Register the CCD response function
    ccd_path = filter_dir / 'DECam_CCD_QE.txt'
    ccd_wave, ccd_trans = np.genfromtxt(ccd_path).T
    ccd_wave_angstroms = ccd_wave * 10  # Convert from nm to Angstroms.
    sncosmo_ccd = sncosmo.Bandpass(ccd_wave_angstroms, ccd_trans)
    sncosmo_ccd.name = 'DECam_ccd'
    sncosmo.register(sncosmo_ccd, force=force)

    # Register the fiducial atmosphere used for the filters
    throughput = Table.read(filter_dir / f'CTIO_DECam.throughput.dat',
                            format='ascii')
    atm = sncosmo.Bandpass(throughput['wave'], throughput['atm'])
    atm.name = 'DECam_atm'
    sncosmo.register(atm, force=force)


def imshow_data(plt_data, z_arr, pwv_arr, figsize=(9, 3), **kwargs):
    """Imshow data for multiple filters, redshifts, and PWVs

    Args:
        z_arr: Array of redshifts
        pwv_arr: Array of PWV values
        figsize: Size of the output figure
        Any arguments for imshow

    Returns:
        - A matplotlib figure
        - An array of matplotlib axes
        - An array of matplotlib colorbar axes
    """

    fig, axes = plt.subplots(
        1, len(plt_data), figsize=figsize, sharex=True, sharey=True)

    # Set default plotting behavior
    default_extent = [min(pwv_arr), max(pwv_arr), min(z_arr), max(z_arr)]
    kwargs.setdefault('extent', default_extent)
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('aspect', 'auto')

    color_bars = []
    for data, axis in zip(plt_data, axes):
        im = axis.imshow(data, **kwargs)

        axis.set_xlabel(r'PWV (mm)')
        axis.xaxis.set_minor_locator(MultipleLocator(1))
        color_bars.append(fig.colorbar(im, ax=axis))

    axes[0].set_ylabel('Redshift')
    axes[0].yaxis.set_minor_locator(MultipleLocator(.1))

    return fig, axes, np.array(color_bars)


def plot_data(plt_data, z_arr, pwv_arr, xaxis='z', figsize=(12, 4), **kwargs):
    """Plot data for multiple filters, redshifts, and PWVs

    Args:
        z_arr: Array of redshifts
        pwv_arr: Array of PWV values
        figsize: Size of the output figure
        Any arguments for plot

    Returns:
        - A matplotlib figure
        - A matplotlib axis
    """

    fig, axes = plt.subplots(
        1, len(plt_data), figsize=figsize, sharex=True, sharey=True)

    if xaxis == 'z':
        for data, axis in zip(plt_data, axes):
            axis.set_xlabel('Redshift')
            axis.set_xlim(min(z_arr), max(z_arr))
            for data_at_pwv, pwv in zip(data.swapaxes(0, 1), pwv_arr):
                axis.plot(z_arr, data_at_pwv, label=f'{pwv:.1f}', **kwargs)

    elif xaxis == 'pwv':
        for data, axis in zip(plt_data, axes):
            axis.set_xlabel('PWV')
            axis.set_xlim(min(pwv_arr), max(pwv_arr))
            for data_at_redshift, z in zip(data, z_arr):
                axis.plot(pwv_arr, data_at_redshift, label=f'{z:.1f}', **kwargs)

    plt.tight_layout()
    return fig, axes
