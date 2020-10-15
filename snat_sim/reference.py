# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Utilities for calculating magnitudes relative to a reference star or
fiducial atmosphere (i.e., fiducial PWV).

Module API
----------
"""

from functools import lru_cache
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pandas as pd

_PARENT = Path(__file__).resolve()
_DATA_DIR = _PARENT.parent.parent / 'data'
_STELLAR_SPECTRA_DIR = _DATA_DIR / 'stellar_spectra'
_STELLAR_FLUX_DIR = _DATA_DIR / 'stellar_fluxes'

available_types = sorted(f.stem for f in _STELLAR_FLUX_DIR.glob('*.txt'))


def _read_stellar_spectra_path(fpath):
    """Load fits file with stellar spectrum from phoenix

    Fits files can be downloaded from:
      http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

    converts from egs/s/cm2/cm to phot/cm2/s/nm using
      https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

    Flux values are returned in phot/cm2/s/angstrom and are index by
    wavelength values in Angstroms.

    Args:
        fpath  (str, Path): Path of the file to read

    Returns:
        Flux values as a pandas Series
    """

    # Load spectral data
    with fits.open(fpath) as infile:
        spec = infile[0].data

    # Load data used to convert spectra to new units
    with fits.open(fpath.parent / 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') as infile:
        lam = infile[0].data  # angstroms

    angstroms_per_cm = 1e8
    conversion_factor = 5.03 * 10 ** 7  # See https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
    ergs_per_photon = conversion_factor * lam

    # Evaluate unit conversion
    spec /= angstroms_per_cm  # ergs/s/cm2/cm into ergs/s/cm2/Angstrom
    spec *= ergs_per_photon  # into phot/cm2/s/angstrom

    indices = (lam >= 3000) & (lam <= 12000)
    return pd.Series(spec[indices], index=lam[indices])


def get_stellar_spectra(spectype):
    """Load spectrum for given spectral type

    Flux values are returned in phot/cm2/s/angstrom and are index by
    wavelength values in Angstroms.

    Args:
        spectype (str): Spectral type (e.g., G2)

    Returns:
        Flux values as a pandas Series
    """

    # Load spectra for different spectral types
    stellar_spectra_dir = _STELLAR_SPECTRA_DIR
    path = next(stellar_spectra_dir.glob(spectype + '*.fits'))
    return _read_stellar_spectra_path(path)


@lru_cache()  # Cache I/O
def get_ref_star_dataframe(reference_type='G2'):
    """Retrieve PWV values to use as reference values

    Args:
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        A DataFrame indexed by PWV with columns for flux
    """

    rpath = _STELLAR_FLUX_DIR / f'{reference_type}.txt'
    if not rpath.exists():
        raise ValueError(
            f'Data not available for specified star {reference_type}. '
            f'Could not find: {rpath}')

    band_names = [f'lsst_hardware_{b}' for b in 'ugrizy']
    column_names = ['PWV'] + band_names
    reference_star_flux = pd.read_csv(
        rpath,
        sep='\s',
        header=None,
        names=column_names,
        comment='#',
        index_col=0,
        engine='python'
    )

    for band in band_names:
        band_flux = reference_star_flux[f'{band}']
        reference_star_flux[f'{band}_norm'] = band_flux / band_flux.loc[0]

    return reference_star_flux


def interp_norm_flux(band, pwv, reference_type='G2'):
    """Return normalized reference star flux values

    Args:
        band           (str): Band to get flux for
        pwv (float, ndarray): PWV values to get magnitudes for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        The normalized flux at the given PWV value(s)
    """

    reference_star_flux = get_ref_star_dataframe(reference_type)
    if np.any(
            (pwv < reference_star_flux.index.min()) |
            (pwv > reference_star_flux.index.max())):
        raise ValueError('PWV is out of range')

    norm_flux = reference_star_flux[f'{band}_norm']
    return np.interp(pwv, norm_flux.index, norm_flux)


def average_norm_flux(band, pwv, reference_types=('G2', 'M5', 'K2')):
    """Return the average normalized reference star flux

    Args:
        band            (str): Band to get flux for
        pwv  (float, ndarray): PWV values to get magnitudes for
        reference_types (str): Types of reference stars to average over

    Returns:
        The normalized flux at the given PWV value(s)
    """

    return np.average([interp_norm_flux(band, pwv, stype) for stype in reference_types])


def divide_ref_from_lc(lc_table, pwv, reference_types=('G2', 'M5', 'K2')):
    """Divide reference flux from a light-curve

    Recalibrate flux values using the average change in flux of a collection of
    reference stars.

    Args:
        lc_table      (Table): Astropy table with columns ``flux`` and ``band``
        pwv           (float): PWV value to subtract reference star for
        reference_types (str): Type of reference stars to use in calibration

    Returns:
        A modified copy of ``lc_table``
    """

    table_copy = lc_table.copy()
    for band in set(table_copy['band']):
        ref_flux = average_norm_flux(band, pwv, reference_types)
        table_copy['flux'][table_copy['band'] == band] /= ref_flux

    return table_copy