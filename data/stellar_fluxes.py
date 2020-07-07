# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tabulate and save to disk stellar flux values through LSST bandpasses at
various PWV concentrations. Much of the tapas related logic (e.g. loading the
TAPAS atmospheric transmissions) could be removed from this module as it is
not used. However, it is kept for reference incase design decisions change
in the future.

This script is heavily based on work undertaken by Ashely Baker.
"""

from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import sncosmo
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from tqdm import tqdm

from sn_analysis import filters
from sn_analysis.transmission import trans_for_pwv

filters.register_lsst_filters()

# Define spectral types this script will consider
SPEC_TYPES = ['G2', 'M4', 'M9', 'M0', 'M1', 'M2', 'M3', 'M5', 'K2', 'K9', 'K5', 'F5']

# Set default data location and PWV sampling
DATA_DIR = Path(__file__).parent
PWV_VALS = np.arange(0, 20, 0.25)


def interp(x, xp, yp, fill=0):
    """Interpolate data using scipy interp1d

    Args:
        x  (ndarray): Points to interpolate for
        xp (ndarray): x values to interpolate from
        yp (ndarray): y values to interpolate from
        fill (float): Fill value for xnew values that are out of range

    Returns:
        Interpolated values as a numpy array
    """

    return interp1d(xp, yp, fill_value=fill, bounds_error=False)(x)


def load_phoenix(fpath, wave_start=750, wave_end=780):
    """Load fits file with stellar spectrum from phoenix

    Fits files can be downloaded from:
      http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

    converts from egs/s/cm2/cm to phot/cm2/s/nm using
      https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

    Args:
        fpath  (str, Path): Path of the file to read
        wave_start (float): Only return fluxes for wavelengths > this value
        wave_end   (float): Only return fluxes for wavelengths < this value

    Returns:
        - An array of wavelength values
        - An array of flux values
    """

    # Load spectral data
    with fits.open(fpath) as infile:
        spec = infile[0].data

    # Load data used to convert spectra to new units
    with fits.open(fpath.parent / 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') as infile:
        lam = infile[0].data  # angstroms

    angstroms_per_cm = 1e8
    angstroms_per_nm = 10
    conversion_factor = 5.03 * 10 ** 7  # See https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
    ergs_per_photon = conversion_factor * lam

    # Evaluate unit conversion
    spec /= angstroms_per_cm  # ergs/s/cm2/cm into ergs/s/cm2/Angstrom
    spec *= ergs_per_photon  # into phot/cm2/s/angstrom
    spec *= angstroms_per_nm  # into phot/cm2/s/nm

    # Return only the requested subarray
    isub = np.where((lam > wave_start * angstroms_per_nm) & (lam < wave_end * angstroms_per_nm))[0]
    return lam[isub] / angstroms_per_nm, spec[isub]


def load_tapas_data(data_dir):
    """Load telluric and filters files

    Loads LSST filters and TAPAS data from the given directory. Data is
    re-sampled to the TAPAS telluric wavelength grid.

    Args:
        data_dir (Path): Directory to read data from

    Returns:
        A dictionary of arrays

    Returned Dictionary Values:
        xtap: Tapas wavelength array
        ytap: Tapas H20 transmission profile
        cv: H20 column density (float)
        yray: Tapas rayleigh scattering profile
        <ugrizy>: LSST filter transmission profiles without the atmospehre
        <G2, M4, ...>: Stellar spectra
    """

    data = {}
    with fits.open(data_dir / 'TAPAS' / 'tapas_kpno_h2o.fits') as tapas:
        # We dont need after 1099 and get nans past there anyways from LSST filters
        data['xtap'] = tapas[1].data['wavelength'][::-1][0:3440000]
        data['ytap'] = tapas[1].data['transmittance'][::-1][0:3440000]
        data['cv'] = float(tapas[1].header['H2OCV'])

    # Load TAPAS rayleigh scattering profile
    rayleigh = fits.getdata(data_dir / 'TAPAS' / 'tapas_rayleigh.fits')
    data['yray'] = interp(
        x=data['xtap'],
        xp=rayleigh['wavelength'][::-1],
        yp=rayleigh['transmittance'][::-1])

    # Load LSST filter profiles
    angstroms_in_nm = 10
    for band_abbrev in 'ugrizy':
        band = sncosmo.get_bandpass(f'lsst_hardware_{band_abbrev}')
        band_wave_in_nm = band.wave / angstroms_in_nm
        data[band_abbrev] = interp(data['xtap'], band_wave_in_nm, band.trans)

    # Load TAPAS spectra for different spectral types
    stel_dir = data_dir / 'stellar_spectra'
    for i, spectype in enumerate(SPEC_TYPES):
        stelname = next(stel_dir.glob(spectype + '*.fits'))
        lam, flux = load_phoenix(stelname, wave_start=500, wave_end=1200)
        data[spectype] = interp(data['xtap'], lam, flux)

    return data


def calculate_flux_pwv(data, spectrum, pwv):
    """Integrate final spectrum after multiplying by rayleigh and H20 absorption

    For more information on the ``data`` argument, see ``load_tapas_data``.

    Args:
        data        (dict): Dictionary of tapas and repository data
        spectrum (ndarray): Spectrum to calculate flux for
        pwv        (float): PWV in mm to calculate flux for

    Returns:
        List of flux values for the LSST ugrizy bands
    """

    # create final spectrum at PWV using tapas (normalize by column density, cv, of template spectrum)
    # airmass=1.0
    # spec_with_trans = spectrum * data['yray'] ** airmass * np.abs(data['ytap']) ** (airmass * pwv / data['cv'])

    # Create final spectrum using pwv_kpno
    spec_with_trans = spectrum * trans_for_pwv(pwv, data['xtap'], resolution=5)

    # Integrate spectrum in each bandpass
    return [trapz(data[b] * spec_with_trans, x=data['xtap']) for b in 'ugrizy']


def calc_flux_pwv_arr(data, pwv_vals):
    """Calculate flux in the LSST ugrizy bands as a function of PWV

    For more information on the ``data`` argument, see ``load_tapas_data``.

    Args:
        data     (dict): Dictionary of tapas and repository data
        pwv_vals (iter): Collection of PWV values to determine fluxes for

    Returns:
        Lists of fluxes in the <ugrizy> bands for each spectral type and PWV
    """

    # initialize flux arrays
    array_size = (len(SPEC_TYPES), len(pwv_vals))
    uflux = np.zeros(array_size)
    gflux = np.zeros(array_size)
    rflux = np.zeros(array_size)
    iflux = np.zeros(array_size)
    zflux = np.zeros(array_size)
    yflux = np.zeros(array_size)

    with tqdm(total=np.product(array_size)) as pbar:
        for i, spectype in enumerate(SPEC_TYPES):
            pbar.set_description(f'Current spectral type {spectype} ({i} / {len(SPEC_TYPES)})')

            for j, pwv in enumerate(pwv_vals):
                uflux[i, j], gflux[i, j], rflux[i, j], iflux[i, j], zflux[i, j], yflux[i, j] = \
                    calculate_flux_pwv(data, data[spectype], pwv)

                pbar.update(1)

    return uflux, gflux, rflux, iflux, zflux, yflux


def run(out_dir, data_dir=DATA_DIR, pwv_vals=PWV_VALS):
    """Sample stellar flux in LSST bandpasses as a function of PWV

    Results are tabulated for multiple spectral types. Each spectral type
    is written to it's own file.

    Args:
        out_dir      (Path): Directory to write output files to
        data_dir     (Path): Project data directory
        pwv_vals (pwv_vals): PWV values to sample for if not default values
    """

    data = load_tapas_data(data_dir)
    uflux, gflux, rflux, iflux, zflux, yflux = calc_flux_pwv_arr(data, pwv_vals)

    # save output
    output_file_header = 'PWV uflux gflux rflux iflux zflux yflux'
    for i, spectype in enumerate(SPEC_TYPES):
        out_path = out_dir / f'{spectype}.txt'
        out_data = np.vstack([PWV_VALS, uflux[i], gflux[i], rflux[i], iflux[i], zflux[i], yflux]).T
        np.savetxt(out_path, out_data, header=output_file_header)


if __name__ == '__main__':
    local_out_dir = Path(__file__).parent / 'stellar_fluxes'
    local_out_dir.mkdir(exist_ok=True)
    run(local_out_dir)
