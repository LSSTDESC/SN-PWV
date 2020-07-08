# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Utilities for calculating magnitudes relative to a reference star or
fiducial atmosphere (i.e., fiducial PWV).
"""

from functools import lru_cache
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import yaml

_PARENT = Path(__file__).resolve()
_DATA_DIR = _PARENT.parent.parent / 'data'
_STELLAR_SPECTRA_DIR = _DATA_DIR / 'stellar_spectra'
_SELLAR_FLUX_DIR = _DATA_DIR / 'stellar_fluxes'

_CONFIG_PATH = _PARENT.parent.parent / 'ref_pwv.yaml'  # Reference pwv values
available_types = sorted(f.stem for f in _SELLAR_FLUX_DIR.glob('*.txt'))


###############################################################################
# Data Parsing and interpolation
###############################################################################


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
def get_config_pwv_vals(config_path=_CONFIG_PATH):
    """Retrieve PWV values to use as reference values

    Returned values include:
        - Lower pwv bound for calculating slope
        - Reference PWV value for normalizing delta m
        - Upper pwv bound for calculating slope

    Args:
        config_path (str): Path of config file if not default

    Returns:
        Dictionary with PWV values in mm
    """

    with open(config_path) as infile:
        config_dict = yaml.load(infile, yaml.BaseLoader)

    return {k: float(v) for k, v in config_dict.items()}


@lru_cache()  # Cache I/O
def get_ref_star_dataframe(reference_type='G2'):
    """Retrieve PWV values to use as reference values

    Args:
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        A DataFrame indexed by PWV with columns for flux
    """

    rpath = _SELLAR_FLUX_DIR / f'{reference_type}.txt'
    if not rpath.exists():
        raise ValueError(
            f'Data not available for specified star {reference_type}. '
            f'Could not find: {rpath}')

    band_abbrevs = 'ugrizy'
    names = ['PWV'] + [f'{b}_flux' for b in band_abbrevs]
    reference_star_flux = pd.read_csv(
        rpath,
        sep='\s',
        header=None,
        names=names,
        comment='#',
        index_col=0,
        engine='python'
    )

    for band in band_abbrevs:
        band_flux = reference_star_flux[f'{band}_flux']
        reference_star_flux[f'{band}_norm'] = band_flux / band_flux.loc[0]

    return reference_star_flux


def interp_norm_flux(band, pwv, reference_type='G2'):
    """Return normalized reference star flux values

    Args:
        band           (str): Band abbreviation <ugrizy> of band to get flux for
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


def interp_norm_mag(band, pwv, reference_type='G2'):
    """Return normalized reference star magnitude

    Interpolation is performed in flux space.

    Args:
        band           (str): Band abbreviation <ugrizy> of band to get flux for
        pwv    (float, list): PWV values to get magnitudes for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        The normalized magnitude at the given PWV value(s)
    """

    return -2.5 * np.log10(interp_norm_flux(band, pwv, reference_type))


###############################################################################
# Subtracting reference magnitudes from various data types
###############################################################################

def divide_ref_from_lc(lc_table, pwv, reference_type='G2'):
    """Divide reference flux from a light-curve

    Args:
        lc_table     (Table): Astropy table with columns ``flux`` and ``band``
        pwv          (float): PWV value to subtract reference star for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        A modified copy of ``lc_table``
    """

    table_copy = lc_table.copy()
    for band in set(table_copy['band']):
        ref_flux = interp_norm_flux(band, pwv, reference_type)
        table_copy['flux'][table_copy['band'] == band] /= ref_flux

    return table_copy


def subtract_ref_star_array(band, norm_mag, pwv, reference_type='G2'):
    """Return reference star magnitudes from an array of normalized magnitudes

    ``pwv_arr`` should be one dimension less than ``norm_mag``

    Args:
        band           (str): Name of the band to subtract magnitudes for
        norm_mag   (ndarray): One or 2d array of magnitudes to subtract from
        pwv    (float, list): PWV values to get magnitudes for
        reference_type (str): Type of reference to use (Default 'G2')

   Returns:
       An array of delta magnitude values
   """

    if np.ndim(norm_mag) - np.ndim(pwv) != 1:
        raise ValueError('``pwv`` should be one dimension less than ``norm_mag``')

    ref_mag = interp_norm_mag(band, pwv, reference_type)
    if np.ndim(ref_mag > 0):
        ref_mag = ref_mag[:, None]

    return norm_mag - ref_mag


def subtract_ref_star_dict(norm_mag, pwv, reference_type='G2'):
    """Determine magnitude relative to a reference star

    Given magnitudes are expected to be normalized relative to the fiducial
    atmosphere.

    Args:
        norm_mag      (dict): Dictionary with arrays of magnitudes for each band
        pwv (float, ndarray): PWV values for each value in norm_mag
        reference_type (str): Type of reference star (Default 'G2')

    Return:
        Dictionary with arrays of magnitudes relative to a reference star for each band
    """

    # Determine normalized magnitude with respect to reference star
    return {band: subtract_ref_star_array(band, norm_mag[band], pwv, reference_type) for band in norm_mag}


def _subtract_ref_star_slope(band, mag_slope, pwv_config, reference_type='G2'):
    """Determine (delta magnitude) / (delta pwv) relative to a reference star

    This function separated from ``subtract_ref_star_slope`` for easier testing

    Args:
        band           (str): Name of the bandpass
        mag_slope  (ndarray): 1d array of slope values
        pwv_config    (dict): Config dictionary for fiducial atmosphere
        reference_type (str): Type of reference star (Default 'G2')

    Return:
        Dictionary with magnitudes relative to a reference star in each band
    """

    # Parse reference pwv values
    pwv_slope_start = pwv_config['slope_start']
    pwv_slope_end = pwv_config['slope_end']

    # Determine slope in normalized magnitude with respect to reference star
    mag_slope_start, mag_slope_end = interp_norm_mag(
        band, [pwv_slope_start, pwv_slope_end], reference_type)

    delta_x = pwv_slope_end - pwv_slope_start
    stellar_delta_y = mag_slope_end - mag_slope_start
    norm_mag_delta_y = mag_slope * delta_x
    return (norm_mag_delta_y - stellar_delta_y) / delta_x


def subtract_ref_star_slope(mag_slope, pwv_config, reference_type='G2'):
    """Determine (delta magnitude) / (delta pwv) relative to a reference star

    Args:
        mag_slope  (ndarray): 1d array of slope values
        pwv_config    (dict): Config dictionary for fiducial atmosphere
        reference_type (str): Type of reference star (Default 'G2')

    Return:
        Dictionary with slopes relative to a reference star in each band
    """

    return {b: _subtract_ref_star_slope(b, mag_slope[b], pwv_config, reference_type) for b in mag_slope}
