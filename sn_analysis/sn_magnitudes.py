# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module determines the change in sn magnitude vs PWV"""

import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import sncosmo
from tqdm import tqdm

from . import modeling

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
config_path = data_dir / 'ref_pwv.yaml'  # Reference pwv values
reference_flux_path = data_dir / 'PWV_absorp/pwv_absorp_type_G2.txt'

# From Betoule 2014
α, β = 0.14, 3.15


###############################################################################
# Determining the PWV induced change in magnitude while leaving all other
# model parameters the same
###############################################################################


def tabulate_mag(source, pwv_arr, z_arr, bands, verbose=True):
    """Calculate apparent magnitude due to presence of PWV
    
    Magnitude is calculated for the model by adding PWV effects
    to a model and leaving all other parameters unchanged.

    Args:
        source      (str): Sncosmo source to use
        pwv_arr (ndarray): Array of PWV values
        z_arr   (ndarray): Array of redshift values
        bands (list[str]): Name of the bands to tabulate magnitudes for
        verbose    (bool): Show a progress bar
        
    Returns:
        A dictionary with 2d arrays for the magnitude at each PWV and redshift
    """

    if verbose:
        iter_total = len(pwv_arr) * len(z_arr) * len(bands)
        pbar = tqdm(total=iter_total, desc='Tabulating Mag')

    model_with_pwv = modeling.get_model_with_pwv(source)
    return_array_shape = (len(pwv_arr), len(z_arr))

    magnitudes = {}
    for band in bands:
        # Performance here is dominated by ``bandmag`` so iteration
        # order is irrelevant. We iterate over bands first for convenience

        mag_arr = []
        for pwv, z in itertools.product(pwv_arr, z_arr):
            model_with_pwv.set(pwv=pwv, z=z)
            mag = model_with_pwv.bandmag(band, 'ab', 0)
            mag_arr.append(mag)

            if verbose:
                pbar.update(1)

        magnitudes[band] = np.reshape(mag_arr, return_array_shape)

    pbar.close()
    return magnitudes


def tabulate_fiducial_mag(source, z_arr, bands, fid_pwv_dict):
    """Get SN magnitudes corresponding to the fiducual atmosphere

    Returns a dictionary of the form
      {<band>: [<slope start mag> , <reference pwv mag>, <slope end mag>]

    Args:
        source        (str): Sncosmo source to use
        z_arr     (ndarray): Array of redshift values
        bands   (list[str]): Name of the bands to tabulate magnitudes for
        fid_pwv_dict (dict): Config dictionary for fiducial atmosphere

    Returns:
        A dictionary with fiducial magnitudes in each band
    """

    # Parse reference pwv values
    pwv_fiducial = fid_pwv_dict['reference_pwv']
    pwv_slope_start = fid_pwv_dict['slope_start']
    pwv_slope_end = fid_pwv_dict['slope_end']

    # Get mag at reference pwv values
    magnitudes = tabulate_mag(
        source=source,
        pwv_arr=[pwv_fiducial, pwv_slope_start, pwv_slope_end],
        z_arr=z_arr,
        bands=bands)

    return magnitudes


###############################################################################
# Determining the PWV induced change in magnitude by simulating light-curves
# with PWV and then fitting a model without a PWV component
###############################################################################


def calibrate_mag(source, mag, params):
    """Calibrate fitted supernova magnitude

    calibrated mag = mag + α * x1 - β * c

    Args:
        source     (str): Source of the model used to fit the given magnitudes
        mag    (ndarray): (n)d array of magnitudes for pwv and redshift
        params (ndarray): (n+1)d array with dimensions for pwv, redshift, parameter

    Returns:
        Array of calibrated magnitudes with same dimensions as ``mag``
    """

    model = sncosmo.Model(source)

    # THe given source must have a stretch and color component
    for param in ('x1', 'c'):
        if param not in model.param_names:
            raise ValueError(
                f'Specified source does not have a ``{param}`` parameter')

    i_x1 = model.param_names.index('x1')
    i_c = model.param_names.index('c')
    return mag + α * params[..., i_x1] - β * params[..., i_c]


def fit_mag(source, light_curves, vparams, pwv_arr, z_arr, bands):
    """Determine apparent mag by fitting simulated light-curves
    
    Returned arrays are shape  (len(pwv_arr), len(z_arr)).
    
    Args:
        source           (str): Sncosmo source to use
        light_curves (ndarray): Array of light-curves to fit
        vparams         (list): Parameters to vary with the fit 
        pwv_arr      (ndarray): Array of PWV values
        z_arr        (ndarray): Array of redshift values
        bands      (list[str]): Name of the bands to tabulate magnitudes for
    
    Returns:
        Dictionary with arrays for fitted magnitude at each PWV and redshift
        Dictionary with arrays for fitted parameters at each PWV and redshift
    """

    # Create model without a PWV parameter
    model_without_pwv = sncosmo.Model(source)

    fitted_mag = {b: [] for b in bands}
    fitted_params = {b: [] for b in bands}
    for lc in light_curves:
        # Use the true light-curve parameters as the initial guess
        lc_parameters = deepcopy(lc.meta)
        lc_parameters.pop('pwv')

        # Fit the model without PWV
        model_without_pwv.update(lc_parameters)
        _, fitted_model = sncosmo.fit_lc(lc, model_without_pwv, vparams)

        for band in bands:
            fitted_mag[band].append(fitted_model.bandmag(band, 'ab', 0))
            fitted_params[band].append(fitted_model.parameters)

    # We could have used a more complicated colloction pattern, but reshaping
    # after the fact is simpler.
    shape = (len(pwv_arr), len(z_arr))
    num_params = len(fitted_model.parameters)
    for band in bands:
        fitted_mag[band] = np.reshape(fitted_mag[band], shape)
        fitted_params[band] = np.reshape(fitted_params[band], (*shape, num_params))

    return fitted_mag, fitted_params


def fit_fiducial_mag(source, obs, vparams, z_arr, bands, fiducial_pwv_dict):
    """Get fitted SN magnitudes corresponding to the fiducual atmosphere

    Returns a dictionary of the form
      {<band>: [<slope start mag> , <reference pwv mag>, <slope end mag>]

    Args:
        source             (str): Sncosmo source to use
        obs              (Table): Array of light-curves to fit
        vparams           (list): Parameters to vary with the fit
        z_arr          (ndarray): Array of redshift values
        band               (str): Name of band to return mag for
        fiducial_pwv_dict (dict): Config dictionary for fiducial atmosphere

    Returns:
        - A dictionary with 2d array of fitted magnitudes in each band
        - A dictionary with 3d array of fitted parameters in each band
    """

    # Parse reference pwv values
    pwv_fiducial = fiducial_pwv_dict['reference_pwv']
    pwv_slope_start = fiducial_pwv_dict['slope_start']
    pwv_slope_end = fiducial_pwv_dict['slope_end']

    # Get mag at reference pwv values
    pwv_vals = [pwv_slope_start, pwv_fiducial, pwv_slope_end]
    light_curves = list(modeling.iter_lcs(obs, source, pwv_vals, z_arr))
    fitted_mag, fitted_params = fit_mag(
        source=source,
        light_curves=light_curves,
        vparams=vparams,
        pwv_arr=pwv_vals,
        z_arr=z_arr,
        bands=bands)

    return fitted_mag, fitted_params


###############################################################################
# Calculating how the values determined above change with PWV
###############################################################################

def calc_delta_mag(mag, fiducial_mag, fiducial_pwv):
    """Determine the change in magnitude relative to the fiducial atmosphere

    This is also equivalent to determining the apparent magnitude of a SN
    normalized to the magnitude at the fiducial atmosphere.

    Args:
        mag          (dict): Dictionary with magnitudes in each band
        fiducial_mag (dict): Dictionary for fiducial atmosphere mag vals
        fiducial_pwv (dict): Dictionary for fiducial atmosphere pwv vals

    Returns:
        - A dictionary with the change in magnitude for each band
        - A dictionary with the slope (mag / pwv) for each band
    """

    # Parse fiducial pwv values
    pwv_slope_start = fiducial_pwv['slope_start']
    pwv_slope_end = fiducial_pwv['slope_end']

    slope = {}
    delta_mag = {}
    for band, (mag_start, mag_fiducial, mag_end) in fiducial_mag.items():
        delta_mag[band] = mag[band] - mag_fiducial

        slope[band] = (
                (mag_end - mag_start) / (pwv_slope_end - pwv_slope_start)
        )

    return delta_mag, slope
