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
_reference_flux_path = data_dir / 'PWV_absorp/pwv_absorp_type_G2.txt'


@lru_cache()  # Cache I/O
def get_config_pwv_vals(config_path=_config_path):
    """Retrieve PWV values t use as reference values

    Returned values include:
        - Lower pwv bound for calculating slope
        - Reference PWV value for normalizing delta m
        - Upper pwv bound for calculating slope

    Args:
        config_path (str): Path of config file if not default

    Returns:
        A list of PWV values in mm
    """

    with open(config_path) as infile:
        config_dict = yaml.load(infile, yaml.BaseLoader)

    return {k: float(v) for k, v in config_dict.items()}


@lru_cache()  # Cache I/O
def get_ref_star_dataframe(reference_path=_reference_flux_path):
    """Retrieve PWV values to use as reference values

    Returned values include:
        - Lower pwv bound for calculating slope
        - Reference PWV value for normalizing delta m
        - Upper pwv bound for calculating slope

    Args:
        reference_path (str): Path of reference stellar flux if not default

    Returns:
        A list of PWV values in mm
    """

    reference_star_flux = pd.read_csv(
        reference_path,
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


def ref_star_mag(band, pwv_arr):
    """Return reference magnitude values as a 2d array

    Args:
        band            (str): Name of the band to get flux for
        pwv_arr     (ndarray): PWV values to get magnitudes for

    Returns:
        Normalized magnitude at the given PWV
    """

    reference_star_flux = get_ref_star_dataframe()
    reference_flux = reference_star_flux.loc[pwv_arr][f'{band}_norm']
    reference_flux = reference_flux[:, None]

    # Since we are using normalized flux
    # m = -2.5log(f / f_atm) + (zp - zp_atm) = -2.5log(f / f_atm)
    normalized_mag = -2.5 * np.log10(reference_flux)
    return normalized_mag


def subtract_ref_star(norm_mag, pwv_arr):
    """Determine magnitude relative to a reference star

    Given magnitudes are expected to be normalized relative to the fiducial
    atmosphere.

    Args:
        band         (str): Name of the bandpass
        norm_mag    (dict): Dict with 2d array of magnitudes for each band
        pwv_arr  (ndarray): PWV values for each magnitude

    Return:
        A 2d array of magnitudes relative to a reference star
    """

    # Determine normalized magnitude with respect to reference star
    return {band: norm_mag[band] - ref_star_mag(band, pwv_arr) for band in norm_mag}


def _subtract_ref_star_slope(band, mag_slope, pwv_config):
    """Determine (delta magnitude) / (delta pwv) relative to a reference star

    Args:
        band          (str): Name of the bandpass
        mag_slope (ndarray): 1d array of slope values

    Return:
        An array of slopes relative to a reference star
    """

    # Parse reference pwv values
    pwv_fiducial = pwv_config['reference_pwv']
    pwv_slope_start = pwv_config['slope_start']
    pwv_slope_end = pwv_config['slope_end']

    # Determine slope in normalized magnitude with respect to reference star
    mag_fiducial, mag_slope_start, mag_slope_end = ref_star_mag(
        band, [pwv_fiducial, pwv_slope_start, pwv_slope_end])

    delta_x = pwv_slope_end - pwv_slope_start
    stellar_delta_y = mag_slope_end - mag_slope_start
    norm_mag_delta_y = mag_slope * delta_x
    return (norm_mag_delta_y - stellar_delta_y) / delta_x


def subtract_ref_star_slope(mag_slope, pwv_config):
    return {b: _subtract_ref_star_slope(b, mag_slope[b], pwv_config) for b in mag_slope}
