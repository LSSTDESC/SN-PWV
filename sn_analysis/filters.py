# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module provides general utilities for plotting data and registering
``sncosmo`` filters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table

FILTER_DIR = Path(__file__).resolve().parent.parent / 'data' / 'filters'


def register_sncosmo_filter(wave: np.ndarray, trans: np.ndarray, name: str, force: bool = False):
    """Register an filter profile with sncosmo

    Args:
        wave: Array of wavelength values in Angstroms
        trans: Array of transmission values between 0 and 1
        name: Name of the filter to register
        force: Whether to overwrite an existing filter with the given name
    """

    # Specifying the name argument in the constructor seems to not work in
    # at least some sncosmo versions, so we set it after instantiation
    sncosmo_ccd = sncosmo.Bandpass(wave, trans)
    sncosmo_ccd.name = name
    sncosmo.register(sncosmo_ccd, force=force)


def register_decam_filters(force: bool = False):
    """Register DECam filter profiles, CCD response, and fiducial ATM with sncosmo

    Args:
        force: Re-register a band if it is already registered
    """

    # Register each filter
    ctio_filter_dir = FILTER_DIR / 'ctio'
    for filter_name in 'ugrizY':
        # Iterate over bands with and without the atmosphere
        for extension in ('', '_filter'):
            band_name = filter_name + extension
            filter_path = ctio_filter_dir / f'CTIO_DECam.{band_name}.dat'

            wave, transmission = np.genfromtxt(filter_path).T
            register_sncosmo_filter(wave, transmission, 'DECam_' + band_name, force)

    # Register the CCD response function
    ccd_path = ctio_filter_dir / 'DECam_CCD_QE.txt'
    ccd_wave, ccd_trans = np.genfromtxt(ccd_path).T
    ccd_wave_angstroms = ccd_wave * 10  # Convert from nm to Angstroms.
    register_sncosmo_filter(ccd_wave_angstroms, ccd_trans, 'DECam_ccd', force)

    # Register the fiducial atmosphere used for the filters
    throughput = Table.read(ctio_filter_dir / f'CTIO_DECam.throughput.dat', format='ascii')
    register_sncosmo_filter(throughput['wave'], throughput['atm'], 'DECam_atm', force)


def register_lsst_filters(force: bool = False):
    """Register DECam filter profiles, CCD response, and fiducial ATM with sncosmo

    Args:
        force: Re-register a band if it is already registered
    """

    lsst_filter_dir = FILTER_DIR / 'lsst_baseline'
    nm_in_angstrom = 10

    # Define file names for each optical component
    bands = 'ugrizy'
    mirrors = range(1, 4)
    file_names = ['detector.dat', 'atmos_10.dat', 'atmos_std.dat']
    file_names.extend(f'filter_{b}.dat' for b in bands)
    file_names.extend(f'hardware_{b}.dat' for b in bands)
    file_names.extend(f'total_{b}.dat' for b in bands)
    file_names.extend(f'm{m}.dat' for m in mirrors)
    file_names.extend(f'lens{m}.dat' for m in mirrors)

    # Read each file into a DataFrame
    response_curves = []
    for fname in file_names:
        file_path = lsst_filter_dir / fname
        wave, trans = np.loadtxt(file_path).T
        wave *= nm_in_angstrom
        register_sncosmo_filter(wave, trans, 'lsst_' + file_path.stem, force)

        # Store filter profiles for later calculations
        filter_series = pd.Series(data=trans, index=wave, name=file_path.stem)
        response_curves.append(filter_series)

    keys = [rc.name for rc in response_curves]
    response_df = pd.concat(response_curves, axis=1, keys=keys)

    # Determine each band without the atmosphere.
    # From the LSST throughput repository README:
    #   total_*.dat throughput curves represent the combination of all
    #   components in the LSST system - mirrors, lenses, filter, detector,
    #   and the zenith atmos_std.dat atmosphere.

    mirrors = response_df.m1 * response_df.m2 * response_df.m3
    register_sncosmo_filter(mirrors.index, mirrors, 'lsst_mirrors', force)

    lenses = response_df.lens1 * response_df.lens2 * response_df.lens3
    register_sncosmo_filter(lenses.index, lenses, 'lsst_lenses', force)

    for band in 'ugrizy':
        filt_name = f'lsst_{band}_no_atm'
        trans = response_df[f'filter_{band}'] * mirrors * lenses * response_df.detector
        register_sncosmo_filter(trans.index, trans, filt_name, force)
