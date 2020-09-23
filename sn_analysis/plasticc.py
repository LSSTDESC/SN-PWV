#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""Read and parse simulated light-curves for different cadences."""

import os
from pathlib import Path
from warnings import warn

import pandas as pd
import sncosmo
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from . import modeling

default_data_dir = Path('/mnt/md0/snsims/')
try:
    plasticc_simulations_directory = Path(os.environ['plasticc_sim_dir'])

except KeyError:
    warn(f'``plasticc_sim_dir`` is not set in environment. Defaulting to {default_data_dir}')
    plasticc_simulations_directory = default_data_dir


def get_available_cadences():
    """Return a list of all available cadences in the PLaSTICC simulation directory"""

    return [p.name for p in plasticc_simulations_directory.glob('*') if p.is_dir()]


def get_model_headers(cadence, model=11):
    """Return a list of all header files for a given cadence and model

    Default is model 11 (Normal SNe)

    Args:
        cadence (str): Name of the cadence to list header files for
        model   (int): Model number to retrieve header paths for

    Returns:
        A list of Path objects
    """

    sim_dir = plasticc_simulations_directory / cadence / f'LSST_WFD_{cadence}_MODEL{model}'
    return list(sim_dir.glob('*HEAD.FITS'))


def iter_lc_for_header(header_path, verbose=True):
    """Iterate over light-curves from a given header file

    Files are expected in pairs of a header file (`*HEAD.fits`) that stores target
    meta data and a photometry file (`*PHOT.fits`) with simulated light-curves.

    Args:
        header_path     (Path, str): Path of the header file
        verbose (bool): Display a progress bar

    Yields:
        - An Astropy table with the MJD and filter for each observation
    """

    # Load meta data from the header file
    with fits.open(header_path) as header_hdulist:
        meta_data = pd.DataFrame(header_hdulist[1].data)
        meta_data = meta_data  # [['PEAKMJD', 'RA', 'DECL', 'SIM_REDSHIFT_CMB', 'PTROBS_MIN', 'PTROBS_MAX']]

    # Load light-curves from the photometry file, This is slow
    phot_file_path = str(header_path).replace('HEAD', 'PHOT')
    with fits.open(phot_file_path) as photometry_hdulist:
        phot_data = Table(photometry_hdulist[1].data)

    # If using pandas instead of astropy on the above line
    # Avoid ValueError: Big-endian buffer not supported on little-endian compiler
    # for key, val in phot_data.iteritems():
    #     phot_data[key] = phot_data[key].to_numpy().byteswap().newbyteorder()

    # phot_data = phot_data[['MJD', 'FLT', 'PHOTFLAG']]
    for idx, meta in tqdm(meta_data.iterrows(), total=len(meta_data), position=1, disable=not verbose):
        lc_start = int(meta['PTROBS_MIN']) - 1
        lc_end = int(meta['PTROBS_MAX'])
        lc = phot_data[lc_start: lc_end]
        lc.meta.update(meta)
        yield lc


def iter_lc_for_cadence_model(cadence, model=11, verbose=True):
    """Iterate over simulated light-curves  for a given cadence

    Args:
        cadence  (str): Name of the cadence to summarize
        model    (int): Model number to retrieve light-curves for
        verbose (bool): Display a progress bar

    Yields:
        An Astropy table with the MJD and filter for each observation
    """

    for header_path in tqdm(get_model_headers(cadence, model), desc=cadence, disable=not verbose):
        for lc in iter_lc_for_header(header_path, verbose):
            yield lc


def format_plasticc_sncosmo(light_curve):
    """Format a PLaSTICC light-curve to be compatible with sncosmo

    Args:
        light_curve (Table): Table of PLaSTICC light-curve data

    Returns:
        An astropy table formatted for use with sncosmo
    """

    lc = Table({
        'time': light_curve['MJD'],
        'band': ['lsst_hardware_' + f.lower().strip() for f in light_curve['FLT']],
        'flux': light_curve['FLUXCAL'],
        'fluxerr': light_curve['FLUXCALERR'],
        'zp': light_curve['ZEROPT'],
        'photflag': light_curve['PHOTFLAG']
    })

    lc['zpsys'] = 'AB'
    lc.meta = light_curve.meta
    return lc


def extract_cadence_data(light_curve, drop_nondetection=False, zp=25, gain=5, skynr=100):
    """Extract the observational cadence from a PLaSTICC light-curve

    Returned table is formatted for use with ``sncosmo.realize_lcs``.

    Args:
        light_curve      (Table): Astropy table with PLaSTICC light-curve data
        drop_nondetection (bool): Drop data with PHOTFLAG == 0
        zp        (float, array): Overwrite the PLaSTICC zero-point with this value
        gain           (int): Gain to use during simulation
        skynr          (int): Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr

    Returns:
        An astropy table with cadence data for the input light-curve
    """

    if drop_nondetection:
        light_curve = light_curve[light_curve['PHOTFLAG'] != 0]

    observations = Table({
        'time': light_curve['MJD'],
        'band': ['lsst_hardware_' + f.lower().strip() for f in light_curve['FLT']],
    })

    observations['zp'] = zp
    observations['zpsys'] = 'ab'
    observations['gain'] = gain
    observations['skynoise'] = light_curve['SKY_SIG'] / skynr
    return observations


def duplicate_plasticc_sncosmo(
        light_curve, source='Salt2-extended', gain=5, skynr=100, scatter=True, cosmo=modeling.betoule_cosmo):
    """Simulate a light-curve with sncosmo that matches the cadence of a PLaSTICC light-curve

    Args:
        light_curve  (Table): Astropy table with PLaSTICC light-curve data
        source (str, Source): Source to use when simulating light-curve flux
        gain           (int): Gain to use during simulation
        skynr          (int): Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr
        scatter       (bool): Add random noise to the flux values
        cosmo    (Cosmology): Rescale the ``x0`` parameter according to the given cosmology

    Returns:
        Astropy table with data for the simulated light-curve
    """

    use_redshift = 'SIM_REDSHIFT_CMB'
    if cosmo is None:
        x0 = light_curve.meta['SIM_SALT2x0']

    else:
        x0 = modeling.calc_x0_for_z(light_curve.meta[use_redshift], 'salt2', cosmo=cosmo)

    params = {
        't0': light_curve.meta['SIM_PEAKMJD'],
        'x1': light_curve.meta['SIM_SALT2x1'],
        'c': light_curve.meta['SIM_SALT2c'],
        'z': light_curve.meta[use_redshift],
        'x0': x0
    }

    observations = extract_cadence_data(light_curve, skynr=skynr, gain=gain)
    return modeling.simulate_lc(observations, source, params, scatter=scatter)
