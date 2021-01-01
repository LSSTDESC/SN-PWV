"""The ``reference_stars`` module includes utilities for calibrating
observed (or simulated) magnitudes relative to a reference star. Reference
star fluxes are determined relative to a fiducial atmosphere with 4mm of PWV.

Usage Example
-------------

The spectrum for individual spectral types can be retreived
using the ``ReferenceStar`` class:

.. doctest:: python

   >>> from snat_sim import reference_stars

   >>> g2_star = reference_stars.ReferenceStar('G2')
   >>> print(g2_star.to_pandas())
   3000.000     4.960049e+17
   3000.006     4.659192e+17
   3000.012     4.304657e+17
   3000.018     3.751426e+17
   3000.024     2.847191e+17
                    ...
   11999.920    1.366567e+18
   11999.940    1.366673e+18
   11999.960    1.366418e+18
   11999.980    1.365863e+18
   12000.000    1.365315e+18
   Length: 933333, dtype: float32


A ``ReferenceCatalog`` is used to represent a collection of stars with different
stellar types. Catalog instances can be used to calibrate supernoca light-curves.

.. code-block:: python

   >>> import sncosmo

   >>> light_curve = sncosmo.load_example_data()
   >>> reference_catalog = reference_stars.ReferenceCatalog('G2', 'M5')
   >>> print(reference_catalog.divide_ref_from_lc(light_curve, pwv=4))

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

from . import constants as const
from ._data_paths import data_paths
from .models import PWVModel

Numeric = Union[int, float]


class ReferenceStar:
    """Representation of spectral data from the Goettingen Spectral Library"""

    def __init__(self, spectral_type: str) -> None:
        """Load a spectrum for the given spectral type

        Flux values are returned in phot/cm2/s/angstrom and are index by
        wavelength values in Angstroms.

        Args:
            spectral_type: Spectral type (e.g., G2)
        """

        self.spectral_type = spectral_type.upper()
        if self.spectral_type not in self.get_available_types():
            raise ValueError(f'Data for spectral type "{self.spectral_type}" is not available.')

        # Load spectra for different spectral types
        path = next(data_paths.stellar_spectra_dir.glob(self.spectral_type + '*.fits'))
        self._spectrum = self._read_stellar_spectra_path(path)

    def to_pandas(self) -> pd.Series:
        """Return the spectral data as a ``pandas.Series`` object"""

        return self._spectrum.copy()

    @staticmethod
    def get_available_types() -> List[str]:
        """Return the spectral types available on disk

        Returns:
            A list fo spectral types
        """

        return sorted(f.stem.upper() for f in data_paths.stellar_flux_dir.glob('*.txt'))

    @staticmethod
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

    @lru_cache()  # Cache I/O
    def get_dataframe(self) -> pd.DataFrame:
        """Retrieve PWV values to use as reference values

        Returns:
            A DataFrame indexed by PWV with columns for flux
        """

        rpath = data_paths.stellar_flux_dir / f'{self.spectral_type}.txt'
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

    def flux(self, band: str, pwv: Union[Numeric, np.array]) -> np.ndarray:
        """Return the reference star flux values

        Args:
            band: Band to get flux for
            pwv: PWV values to get magnitudes for

        Returns:
            The normalized flux at the given PWV value(s)
        """

        reference_star_flux = self.get_dataframe()
        if np.any(
                (pwv < reference_star_flux.index.min()) |
                (pwv > reference_star_flux.index.max())):
            raise ValueError('PWV is out of range')

        norm_flux = reference_star_flux[band]
        return np.interp(pwv, norm_flux.index, norm_flux)

    def norm_flux(self, band: str, pwv: Union[Numeric, np.array]) -> np.ndarray:
        """Return the normalized reference star flux values

        Args:
            band: Band to get flux for
            pwv: PWV values to get magnitudes for

        Returns:
            The normalized flux at the given PWV value(s)
        """

        reference_star_flux = self.get_dataframe()
        if np.any(
                (pwv < reference_star_flux.index.min()) |
                (pwv > reference_star_flux.index.max())):
            raise ValueError('PWV is out of range')

        norm_flux = reference_star_flux[f'{band}_norm']
        return np.interp(pwv, norm_flux.index, norm_flux)


class ReferenceCatalog:
    """A rudimentary implementation of a reference star catalog"""

    def __init__(self, *spectral_types: str) -> None:
        """Create a reference star catalog composed of the given spectral types

        Args:
            *spectral_types: Spectral types for the catalog (e.g., 'G2', 'M5', 'K2')
        """

        if not spectral_types:
            raise ValueError('Must specify at least one spectral type for the catalog.')

        self.spectral_types = spectral_types
        self.spectra = tuple(ReferenceStar(st) for st in spectral_types)

    def average_norm_flux(self, band: str, pwv: Union[Numeric, Collection, np.ndarray], ) -> np.ndarray:
        """Return the average normalized reference star flux

        Args:
            band: Band to get flux for
            pwv: PWV values to get magnitudes for

        Returns:
            The normalized flux at the given PWV value(s)
        """

        return np.average([s.norm_flux(band, pwv) for s in self.spectra], axis=0)

    def divide_ref_from_lc(self, lc_table: Table, pwv: Union[Numeric, np.ndarray]) -> Table:
        """Divide reference flux from a light-curve

        Recalibrate flux values using the average change in flux of a collection of
        reference stars.

        Args:
            lc_table: Astropy table with columns ``flux`` and ``band``
            pwv: PWV value to subtract reference star for

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
            table_copy['flux'][band_indices] /= self.average_norm_flux(band, pwv[band_indices])

        return table_copy


class VariableCatalog:
    """A reference star catalog that determines the time dependent PWV concentration from an underlying PWV model"""

    def __init__(self, *spectral_types: str, pwv_model: PWVModel) -> None:
        """Create a reference star catalog composed of the given spectral types and a PWV model

        Args:
            *spectral_types: Spectral types for the catalog (e.g., 'G2', 'M5', 'K2')
            pwv_model: The PWV model to determine the zenith PWV concentration from
        """

        self.catalog = ReferenceCatalog(*spectral_types)
        self.pwv_model = pwv_model

    def average_norm_flux(
            self,
            band: str,
            time: Union[float, np.ndarray, Collection],
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> np.ndarray:
        """Return the average normalized reference star flux

        Args:
            band: Band to get flux for
            time: Time at which the target is observed
            ra: Right Ascension of the target (Deg)
            dec: Declination of the target (Deg)
            lat: Latitude of the observer (Deg)
            lon: Longitude of the observer (Deg)
            alt: Altitude of the observer (m)
            time_format: Astropy supported format of the time value (Default: 'mjd')

        Returns:
            The normalized flux at the given PWV value(s)
        """

        pwv = self.pwv_model.pwv_los(time, ra, dec, lat, lon, alt, time_format=time_format)
        return self.catalog.average_norm_flux(band, pwv)

    def divide_ref_from_lc(
            self,
            lc_table: Table,
            time: Union[float, np.ndarray, Collection],
            ra: float,
            dec: float,
            lat: float = const.vro_latitude,
            lon: float = const.vro_longitude,
            alt: float = const.vro_altitude,
            time_format: str = 'mjd'
    ) -> Table:
        """Divide reference flux from a light-curve

        Recalibrate flux values using the average change in flux of a collection of
        reference stars.

        Args:
            lc_table: Astropy table with columns ``flux`` and ``band``
            time: Time at which the target is observed
            ra: Right Ascension of the target (Deg)
            dec: Declination of the target (Deg)
            lat: Latitude of the observer (Deg)
            lon: Longitude of the observer (Deg)
            alt: Altitude of the observer (m)
            time_format: Astropy supported format of the time value (Default: 'mjd')

        Returns:
            A modified copy of ``lc_table``
        """

        pwv = self.pwv_model.pwv_los(time, ra, dec, lat, lon, alt, time_format=time_format)
        return self.catalog.divide_ref_from_lc(lc_table, pwv)
