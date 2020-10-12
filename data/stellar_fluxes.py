# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tabulate and save to disk stellar flux values through LSST bandpasses at
various PWV concentrations.

This script is based on work undertaken by Ashely Baker.
"""

from pathlib import Path

import numpy as np
import sncosmo
from scipy.integrate import trapz
from tqdm import tqdm

from sn_analysis import filters
from sn_analysis.reference import get_stellar_spectra
from pwv_kpno.defaults import v1_transmission

filters.register_lsst_filters()

# Define spectral types this script will consider
SPEC_TYPES = ('G2', 'M4', 'M9', 'M0', 'M1', 'M2', 'M3', 'M5', 'K2', 'K9', 'K5', 'F5')

# Set default data location and PWV sampling
DATA_DIR = Path(__file__).parent
PWV_VALS = np.arange(0, 20, 0.5)


def calculate_lsst_fluxes(spectrum, pwv):
    """Integrate final spectrum after multiplying by rayleigh and H20 absorption

    For more information on the ``data`` argument, see ``get_stellar_spectra``.

    Args:
        spectrum (Series): Spectrum to calculate flux for
        pwv       (float): PWV in mm to calculate flux for

    Returns:
        List of flux values for the LSST ugrizy bands
    """

    transmission = v1_transmission(pwv, spectrum.index, res=5)
    spec_with_trans = spectrum * transmission

    # Integrate spectrum in each bandpass
    fluxes = []
    for band_abbrev in 'ugrizy':
        band = sncosmo.get_bandpass(f'lsst_hardware_{band_abbrev}')
        band_transmission = band(spec_with_trans.index)
        flux = trapz(band_transmission * spec_with_trans, x=spec_with_trans.index)
        fluxes.append(flux)

    return fluxes


def run(out_dir, spec_types=SPEC_TYPES, pwv_vals=PWV_VALS):
    """Sample stellar flux in LSST bandpasses as a function of PWV

    Results are tabulated for multiple spectral types. Each spectral type
    is written to it's own file.

    Args:
        out_dir    (Path): Directory to write output files to
        spec_types (iter): Spectral types to tabulate values for
        pwv_vals   (iter): PWV values to sample for if not default values
    """

    output_file_header = 'PWV uflux gflux rflux iflux zflux yflux'
    total_iters = len(spec_types) * len(pwv_vals)
    with tqdm(total=total_iters) as pbar:
        for i, spectype in enumerate(spec_types):
            out_path = out_dir / f'{spectype}.txt'
            spectrum = get_stellar_spectra(spectype)
            pbar.set_description(f'Current spectral type {spectype} ({i} / {len(spec_types)})')

            spectrum_flux = np.zeros((len(pwv_vals), 7))
            spectrum_flux[:, 0] = pwv_vals
            for i, pwv in enumerate(pwv_vals):
                spectrum_flux[i, 1:] = calculate_lsst_fluxes(spectrum, pwv)
                pbar.update(1)

            np.savetxt(out_path, spectrum_flux, header=output_file_header)


if __name__ == '__main__':
    local_out_dir = Path(__file__).parent / 'stellar_fluxes'
    local_out_dir.mkdir(exist_ok=True)
    run(local_out_dir)
