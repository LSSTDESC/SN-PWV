# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Utilities for calculating magnitudes relative to a reference star or
fiducial atmosphere (i.e., fiducial PWV).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_file = Path(__file__).resolve()
data_dir = _file.parent.parent / 'data'
_config_path = _file.parent.parent / 'ref_pwv.yaml'  # Reference pwv values
_stellar_type_paths = {
    'F5': data_dir / 'PWV_absorp/pwv_absorp_type_F5.txt',
    'G2': data_dir / 'PWV_absorp/pwv_absorp_type_G2.txt',
    'M4': data_dir / 'PWV_absorp/pwv_absorp_type_M4.txt',
    'M9': data_dir / 'PWV_absorp/pwv_absorp_type_M9.txt'
}


###############################################################################
# Data Parsing
###############################################################################

@lru_cache()  # Cache I/O
def get_config_pwv_vals(config_path=_config_path):
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

    try:
        rpath = _stellar_type_paths[reference_type]

    except KeyError:
        keys = list(_stellar_type_paths.keys())
        raise ValueError(
            f'Data not available for specified star {reference_type}. '
            f'available options include: {keys}')

    reference_star_flux = pd.read_csv(
        rpath,
        sep='\s',
        header=None,
        names=['PWV', 'decam_z_flux', 'decam_i_flux', 'decam_r_flux'],
        comment='#',
        index_col=0,
        engine='python'
    )

    ref_pwv = get_config_pwv_vals()['reference_pwv']
    for band in 'riz':
        band_flux = reference_star_flux[f'decam_{band}_flux']
        reference_star_flux[f'decam_{band}_norm'] = band_flux / band_flux.loc[ref_pwv]

    return reference_star_flux


def ref_star_norm_flux(band, pwv, reference_type='G2'):
    """Return normalized reference star flux values

    Args:
        band           (str): Name of the band to get flux for
        pwv (float, ndarray): PWV values to get magnitudes for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        The normalized flux at the given PWV value(s)
    """

    reference_star_flux = get_ref_star_dataframe(reference_type)
    reference_flux = reference_star_flux.loc[pwv][f'{band}_norm']
    if isinstance(reference_flux, pd.Series):
        reference_flux = np.array(reference_flux)

    return reference_flux


def ref_star_norm_mag(band, pwv, reference_type='G2'):
    """Return normalized reference star magnitude values as a 2d array

    Args:
        band           (str): Name of the band to get flux for
        pwv    (float, list): PWV values to get magnitudes for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        The normalized magnitude at the given PWV value(s)
    """

    return -2.5 * np.log10(ref_star_norm_flux(band, pwv, reference_type))


###############################################################################
# Subtracting reference magnitudes from various data models
###############################################################################

def subtract_ref_from_lc(lc_table, pwv, reference_type='G2'):
    """Subtract reference flux from a light-curve

    Args:
        lc_table     (Table): Astropy table with columns ``flux`` and ``band``
        pwv          (float): PWV value to subtract reference star for
        reference_type (str): Type of reference star (Default 'G2')

    Returns:
        A modified copy of ``lc_table``
    """

    table_copy = lc_table.copy()
    for band in set(table_copy['band']):
        ref_flux = ref_star_norm_flux(band, pwv, reference_type)
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

    ref_mag = ref_star_norm_mag(band, pwv, reference_type)
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
    mag_slope_start, mag_slope_end = ref_star_norm_mag(
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
