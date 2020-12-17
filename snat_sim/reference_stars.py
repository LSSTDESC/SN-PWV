"""The ``reference_stars`` module includes utilities for calibrating
observed (or simulated) magnitudes relative to a reference star. Reference
star fluxes are determined relative to a fiducial atmosphere with 4mm of PWV.

Module Docs
-----------
"""

from functools import lru_cache
from pathlib import Path
from typing import *
from typing import Collection

import astropy.io.fits as fits
import numpy as np
import pandas as pd
from astropy.table import Table

from ._data_paths import data_paths

Numeric = Union[int, float]


def get_available_types() -> List[str]:
    """Return the spectral types available on disk

    Returns:
        A list fo spectral types
    """

    return sorted(f.stem for f in data_paths.stellar_flux_dir.glob('*.txt'))


def _read_stellar_spectra_path(fpath: Union[str, Path]) -> pd.Series:
    """Load fits file with stellar spectrum from phoenix

    Fits files can be downloaded from:
      http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

    converts from egs/s/cm2/cm to phot/cm2/s/nm using
      https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

    Flux values are returned in phot/cm2/s/angstrom and are index by
    wavelength values in Angstroms.

    Args:
        fpath: Path of the file to read

    Returns:
        Flux values as a pandas Series
    """

    # Load spectral data
    with fits.open(fpath) as infile:
        spec = infile[0].data

    # Load data used to convert spectra to new units
    # noinspection SpellCheckingInspection
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


def get_stellar_spectra(spectral_type: str) -> pd.Series:
    """Load spectrum for given spectral type

    Flux values are returned in phot/cm2/s/angstrom and are index by
    wavelength values in Angstroms.

    Args:
        spectral_type: Spectral type (e.g., G2)

    Returns:
        Flux values as a pandas Series
    """

    # Load spectra for different spectral types
    path = next(data_paths.stellar_spectra_dir.glob(spectral_type + '*.fits'))
    return _read_stellar_spectra_path(path)


@lru_cache()  # Cache I/O
def get_ref_star_dataframe(spectral_type: str = 'G2') -> pd.DataFrame:
    """Retrieve PWV values to use as reference values

    Args:
        spectral_type: Type of reference star (Default 'G2')

    Returns:
        A DataFrame indexed by PWV with columns for flux
    """

    rpath = data_paths.stellar_flux_dir / f'{spectral_type}.txt'
    if not rpath.exists():
        raise ValueError(
            f'Data not available for specified star {spectral_type}. '
            f'Could not find: {rpath}')

    band_names = [f'lsst_hardware_{b}' for b in 'ugrizy']
    column_names = ['PWV'] + band_names
    reference_star_flux = pd.read_csv(
        rpath,
        sep='\\s',
        header=None,
        names=column_names,
        comment='#',
        index_col=0,
        engine='python'
    )

    for band in band_names:
        band_flux = reference_star_flux[f'{band}']
        reference_star_flux[f'{band}_norm'] = band_flux / band_flux.loc[0]

    return reference_star_flux


def interp_norm_flux(band: str, pwv: Union[Numeric, np.array], spectral_type='G2') -> np.ndarray:
    """Return normalized reference star flux values

    Args:
        band: Band to get flux for
        pwv: PWV values to get magnitudes for
        spectral_type: Type of reference star (Default 'G2')

    Returns:
        The normalized flux at the given PWV value(s)
    """

    reference_star_flux = get_ref_star_dataframe(spectral_type)
    if np.any(
            (pwv < reference_star_flux.index.min()) |
            (pwv > reference_star_flux.index.max())):
        raise ValueError('PWV is out of range')

    norm_flux = reference_star_flux[f'{band}_norm']
    return np.interp(pwv, norm_flux.index, norm_flux)


def average_norm_flux(
        band: str, pwv: Union[Numeric, np.ndarray], spectral_types: Collection[str] = ('G2', 'M5', 'K2')
) -> np.ndarray:
    """Return the average normalized reference star flux

    Args:
        band: Band to get flux for
        pwv: PWV values to get magnitudes for
        spectral_types: Types of reference stars to average over

    Returns:
        The normalized flux at the given PWV value(s)
    """

    return np.average([interp_norm_flux(band, pwv, stype) for stype in spectral_types], axis=0)


def divide_ref_from_lc(
        lc_table: Table, pwv: Union[Numeric, np.ndarray], spectral_types: Collection[str] = ('G2', 'M5', 'K2')
) -> Table:
    """Divide reference flux from a light-curve

    Recalibrate flux values using the average change in flux of a collection of
    reference stars.

    Args:
        lc_table: Astropy table with columns ``flux`` and ``band``
        pwv: PWV value to subtract reference star for
        spectral_types: Type of reference stars to use in calibration

    Returns:
        A modified copy of ``lc_table``
    """

    if isinstance(pwv, Collection):
        if len(pwv) != len(lc_table):
            raise ValueError('PWV must be a float or have the same length as ``lc_table``')

        pwv = np.array(pwv)

    else:
        pwv = np.full(len(lc_table), pwv)

    table_copy = lc_table.copy()
    for band in set(table_copy['band']):
        band_indices = np.where(table_copy['band'] == band)[0]
        table_copy['flux'][band_indices] /= average_norm_flux(band, pwv[band_indices], spectral_types)

    return table_copy
