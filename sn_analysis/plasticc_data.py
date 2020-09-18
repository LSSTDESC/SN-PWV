#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""Read and parse simulated light-curves for different cadences."""

import os
from pathlib import Path
from warnings import warn

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

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


def iter_lc_for_header(header_path):
    """Iterate over light-curves from a given header file

    Files are expected in pairs of a header file (`*HEAD.fits`) that stores target
    meta data and a photometry file (`*PHOT.fits`) with simulated light-curves.

    Args:
        header_path     (Path, str): Path of the header file

    Yields:
        - An Astropy table with the MJD and filter for each observation
    """

    # Load meta data from the header file
    with fits.open(header_path) as header_hdulist:
        meta_data = pd.DataFrame(header_hdulist[1].data)
        meta_data = meta_data  #[['PEAKMJD', 'RA', 'DECL', 'SIM_REDSHIFT_CMB', 'PTROBS_MIN', 'PTROBS_MAX']]

    # Load light-curves from the photometry file, This is slow
    phot_file_path = str(header_path).replace('HEAD', 'PHOT')
    with fits.open(phot_file_path) as photometry_hdulist:
        phot_data = Table(photometry_hdulist[1].data)

    # If using pandas instead of astropy on the above line
    # Avoid ValueError: Big-endian buffer not supported on little-endian compiler
    # for key, val in phot_data.iteritems():
    #     phot_data[key] = phot_data[key].to_numpy().byteswap().newbyteorder()

    # phot_data = phot_data[['MJD', 'FLT', 'PHOTFLAG']]
    for idx, meta in meta_data.iterrows():
        lc_start = int(meta['PTROBS_MIN']) - 1
        lc_end = int(meta['PTROBS_MAX'])
        lc = phot_data[lc_start: lc_end]
        lc.meta.update(meta)
        yield lc


def iter_lc_for_cadence_model(cadence, model=11, verbose=True):
    """Iterate over target pointings  for a given cadence

    Args:
        cadence (str): Name of the cadence to summarize
        model   (int): Model number to retrieve light-curves for

    Yields:
        - An Astropy table with the MJD and filter for each observation
    """

    for header_path in tqdm(get_model_headers(cadence, model), desc=cadence, disable=~verbose):
        for lc in iter_lc_for_header(header_path):
            yield lc

import sncosmo
sncosmo.realize_lcs()