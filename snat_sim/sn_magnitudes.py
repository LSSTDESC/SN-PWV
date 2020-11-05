"""The ``sn_magnitudes.py`` module is responsible for calculating supernova
magnitudes as a function of PWV and redshift. Functionality is provided to
determine magnitudes directly from a SN model and via a light-curve fit.

Module API
----------
"""

import itertools
from copy import copy
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import sncosmo
import yaml
from tqdm import tqdm

from . import modeling, constants as const

# Reference pwv values
_CONFIG_PATH = Path(__file__).resolve().parent / 'defaults' / 'ref_pwv.yaml'


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


###############################################################################
# Determining the PWV induced change in magnitude while leaving all other
# model parameters the same
###############################################################################


def tabulate_mag(model, pwv_arr, z_arr, bands, verbose=True):
    """Calculate apparent magnitude due to presence of PWV
    
    Magnitude is calculated for the model by adding PWV effects
    to a model and leaving all other parameters unchanged.

    Args:
        model     (Model): The sncosmo model to use in the simulations
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

    return_array_shape = (len(pwv_arr), len(z_arr))

    magnitudes = {}
    for band in bands:
        # Performance here is dominated by ``bandmag`` so iteration
        # order is irrelevant. We iterate over bands first for convenience

        mag_arr = []
        for pwv, z in itertools.product(pwv_arr, z_arr):
            model.set(pwv=pwv, z=z, x0=modeling.calc_x0_for_z(z, model.source))
            mag = model.bandmag(band, 'ab', 0)
            mag_arr.append(mag)

            if verbose:
                # noinspection PyUnboundLocalVariable
                pbar.update()

        magnitudes[band] = np.reshape(mag_arr, return_array_shape)

    if verbose:
        pbar.close()

    return magnitudes


def tabulate_fiducial_mag(model, z_arr, bands, fid_pwv_dict=None):
    """Get SN magnitudes corresponding to the fiducial atmosphere

    Returns a dictionary of the form
      {<band>: [<slope start mag> , <reference pwv mag>, <slope end mag>]

    Args:
        model       (Model): The sncosmo model to use in the simulations
        z_arr     (ndarray): Array of redshift values
        bands   (list[str]): Name of the bands to tabulate magnitudes for
        fid_pwv_dict (dict): Config dictionary for fiducial atmosphere

    Returns:
        A dictionary with fiducial magnitudes in each band
    """

    if fid_pwv_dict is None:
        fid_pwv_dict = get_config_pwv_vals()

    # Parse reference pwv values
    pwv_fiducial = fid_pwv_dict['reference_pwv']
    pwv_slope_start = fid_pwv_dict['slope_start']
    pwv_slope_end = fid_pwv_dict['slope_end']

    # Get mag at reference pwv values
    magnitudes = tabulate_mag(
        model=model,
        pwv_arr=[pwv_slope_start, pwv_fiducial, pwv_slope_end],
        z_arr=z_arr,
        bands=bands)

    return magnitudes


###############################################################################
# Determining the PWV induced change in magnitude by simulating light-curves
# with PWV and then fitting a model without a PWV component
###############################################################################


def correct_mag(model, mag, params, alpha=const.betoule_alpha, beta=const.betoule_beta):
    """Correct fitted supernova magnitude for stretch and color

    calibrated mag = mag + α * x1 - β * c

    Args:
        model    (Model): Model used to fit the given magnitudes
        mag    (ndarray): (n)d array of magnitudes for pwv and redshift
        params (ndarray): (n+1)d array with dimensions for pwv, redshift, parameter
        alpha    (float): Alpha parameter value
        beta     (float): Beta parameter value

    Returns:
        Array of calibrated magnitudes with same dimensions as ``mag``
    """

    # THe given model must have a stretch and color component
    for param in ('x1', 'c'):
        if param not in model.param_names:
            raise ValueError(
                f'Specified model does not have a ``{param}`` parameter')

    i_x1 = model.param_names.index('x1')
    i_c = model.param_names.index('c')
    return mag + alpha * params[..., i_x1] - beta * params[..., i_c]


def fit_mag(model, light_curves, vparams, bands, pwv_arr=None, z_arr=None, **kwargs):
    """Determine apparent mag by fitting simulated light-curves
    
    Returned arrays are shape  (len(pwv_arr), len(z_arr)).
    
    Args:
        model          (Model): The sncosmo model to use when fitting
        light_curves (ndarray): Array of light-curves to fit
        vparams         (list): Parameters to vary with the fit 
        pwv_arr      (ndarray): Array of PWV values
        z_arr        (ndarray): Array of redshift values
        bands      (list[str]): Name of the bands to tabulate magnitudes for
        Any arguments for ``sncosmo.fit_lc``.
    
    Returns:
        Dictionary with arrays for fitted magnitude at each PWV and redshift
        Dictionary with arrays for fitted parameters at each PWV and redshift
    """

    model = copy(model)

    fitted_mag = {b: [] for b in bands}
    fitted_params = {b: [] for b in bands}
    for lc in light_curves:
        # Use the true light-curve parameters as the initial guess
        lc_parameters = deepcopy(lc.meta)
        lc_parameters.pop('pwv', None)
        lc_parameters.pop('res', None)

        # Fit the model without PWV
        model.update(lc_parameters)
        _, fitted_model = sncosmo.fit_lc(lc, model, vparams, **kwargs)

        for band in bands:
            fitted_mag[band].append(fitted_model.bandmag(band, 'ab', 0))
            fitted_params[band].append(fitted_model.parameters)

    if pwv_arr is not None and z_arr is not None:
        # We could have used a more complicated collection pattern, but reshaping
        # after the fact is simpler.
        shape = (len(pwv_arr), len(z_arr))
        num_params = len(fitted_model.parameters)
        for band in bands:
            fitted_mag[band] = np.reshape(fitted_mag[band], shape)
            fitted_params[band] = np.reshape(fitted_params[band], (*shape, num_params))

    return fitted_mag, fitted_params


def fit_fiducial_mag(sim_model, fit_model, obs, vparams, z_arr, bands, fid_pwv_dict=None):
    """Get fitted SN magnitudes corresponding to the fiducial atmosphere

    Returns a dictionary of the form
      {<band>: [<slope start mag> , <reference pwv mag>, <slope end mag>]

    Args:
        sim_model        (Model): The sncosmo model to use when simulating light-curves
        fit_model        (Model): The sncosmo model to use when fitting
        obs              (Table): Array of light-curves to fit
        vparams           (list): Parameters to vary with the fit
        z_arr          (ndarray): Array of redshift values
        bands        (list[str]): Name of band to return mag for
        fid_pwv_dict (dict): Config dictionary for fiducial atmosphere

    Returns:
        - A dictionary with 2d array of fitted magnitudes in each band
        - A dictionary with 3d array of fitted parameters in each band
    """

    if fid_pwv_dict is None:
        fid_pwv_dict = get_config_pwv_vals()

    # Parse reference pwv values
    pwv_fiducial = fid_pwv_dict['reference_pwv']
    pwv_slope_start = fid_pwv_dict['slope_start']
    pwv_slope_end = fid_pwv_dict['slope_end']

    # Get mag at reference pwv values
    pwv_vals = [pwv_slope_start, pwv_fiducial, pwv_slope_end]
    light_curves = modeling.iter_lcs(obs, sim_model, pwv_vals, z_arr)
    fitted_mag, fitted_params = fit_mag(
        model=fit_model,
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


###############################################################################
# Distance Modulus Calculations
###############################################################################

def calc_mu_for_model(model, cosmo=const.betoule_cosmo):
    """Calculate the distance modulus of a model

    Args:
        model     (Model): An sncosmo model
        cosmo (Cosmology): Cosmology to use in the calculation

    Returns:
        mu = m_B - M_B
    """

    dilation_factor = 1 + model['z']
    time_dilation_mag_offset = - 2.5 * np.log10(dilation_factor)

    b_band = sncosmo.get_bandpass('standard::b')
    rest_band = b_band.shifted(dilation_factor)

    apparent_mag = model.bandmag(rest_band, 'ab', 0) - time_dilation_mag_offset
    absolute_mag = model.source_peakabsmag(b_band, 'ab', cosmo=cosmo)
    return apparent_mag - absolute_mag


def calc_mu_for_params(model, params):
    """Calculate the distance modulus for an array of fitted params

    Args:
        model    (Model): The sncosmo model to use in the simulations
        params (ndarray): n-dimensional array of parameters

    Returns:
        An array of distance moduli with one dimension less than ``params``
    """

    param_shape = np.shape(params)[:-1]
    num_param_arrays = np.prod(param_shape)
    num_params = params.shape[-1]

    reshaped_params = np.reshape(params, (num_param_arrays, num_params))

    mu = []
    model = copy(model)
    for param_arr in reshaped_params:
        model.parameters = param_arr  # We don't need to use `update` because `parameters` has a setter
        mu.append(calc_mu_for_model(model))

    return np.reshape(mu, param_shape)


def calc_calibration_factor_for_params(model, params):
    """Calculate the distance modulus for an array of fitted params

    returns constants.alpha * x_1 - constants.beta * c

    Args:
        model          (Model): The sncosmo model to use in the simulations
        params (ndarray): n-dimensional array of parameters

    Returns:
        An array of calibration factors with one dimension less than ``params``
    """

    params_dict = {
        param: params[..., i] for
        i, param in enumerate(model.param_names)
    }

    return const.betoule_alpha * params_dict['x1'] - const.betoule_beta * params_dict['c']
