# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""This module provides general utilities for plotting data and registering
``sncosmo`` filters.
"""

from pathlib import Path

import numpy as np
import sncosmo
from astropy.table import Table

filter_dir = Path(__file__).resolve().parent.parent / 'data' / 'filters'


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
    throughput = Table.read(filter_dir / f'CTIO_DECam.throughput.dat', format='ascii')
    atm = sncosmo.Bandpass(throughput['wave'], throughput['atm'])
    atm.name = 'DECam_atm'
    sncosmo.register(atm, force=force)
