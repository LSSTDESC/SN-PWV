"""The ``plasticc`` module provides data access for locally available
PLaSTICC simulations. Data is accessible by specifying the cadence and
model used in a given simulation.

Usage Example
-------------

The ``plasticc`` module makes it easy to check what data is available in
the current working environment:

.. doctest:: python

   >>> from snat_sim import plasticc

   >>> # Check where the `snat_sim` package is expecting to find data
   >>> print(plasticc.get_data_dir())  #doctest:+SKIP

   >>> # Get a list of cadences available in the directory printed above
   >>> print(plasticc.get_available_cadences())  #doctest:+SKIP

   >>> # Count the number of light-curves for a given cadence and SN model
   >>> num_lc = plasticc.count_light_curves('alt_sched', model=11)

It also provides **basic** data access via the construction of an iterator
over all available light-curves for a given cadence / model. You should expect
the first evaluation of the iterator to be slow since it has to load
light-curve data into memory as chunks.

.. code-block:: python

   >>> lc_iterator = plasticc.iter_lc_for_cadence_model('alt_sched', model=11)
   >>> plasticc_lc = next(lc_iterator)

PLaSTICC simulations were run using the ``SNANA`` package in FORTRAN and thus
are returned using the ``SNANA`` data model. Alternatively, you can convert the
returned tables to the data model used by the ``sncosmo`` Python package.

.. code-block:: python

   >>> formatted_lc = plasticc.format_plasticc_sncosmo(plasticc_lc)

Module Docs
-----------
"""

from pathlib import Path
from typing import *

import pandas as pd
from astropy.cosmology.core import Cosmology
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from snat_sim import constants as const
from ._data_paths import data_paths


def get_available_cadences() -> List[str]:
    """Return a list of all available cadences in the PLaSTICC simulation directory"""

    return [p.name for p in data_paths.get_plasticc_dir().glob('*') if p.is_dir()]


def get_model_headers(cadence: str, model: int) -> List[Path]:
    """Return a list of all header files for a given cadence and model

    Default is model 11 (Normal SNe)

    Args:
        cadence: Name of the cadence to list header files for
        model: Model number to retrieve header paths for

    Returns:
        A list of Path objects
    """

    return list(data_paths.get_plasticc_dir(cadence, model).glob('*HEAD.FITS'))


def count_light_curves(cadence: str, model: int) -> int:
    """Return the number of available light-curve simulations for a given cadence and model

    Args:
        cadence: Name of the cadence to list header files for
        model: Model number to retrieve header paths for

    Returns:
        Number of simulated light-curves available in the working environment
    """

    cadence_header_files = get_model_headers(cadence, model)

    total_lc = 0
    for header_path in cadence_header_files:
        with fits.open(header_path) as _temp:
            total_lc += len(_temp[1].data)

    return total_lc


def iter_lc_for_header(header_path: Union[Path, str], verbose: bool = True):
    """Iterate over light-curves from a given header file

    Files are expected in pairs of a header file (`*HEAD.fits`) that stores target
    meta data and a photometry file (`*PHOT.fits`) with simulated light-curves.

    Args:
        header_path: Path of the header file
        verbose: Display a progress bar

    Yields:
        An Astropy table with the MJD and filter for each observation
    """

    # Load meta data from the header file
    with fits.open(header_path) as header_hdulist:
        meta_data = pd.DataFrame(header_hdulist[1].data)

    # Load light-curves from the photometry file, This is slow
    phot_file_path = str(header_path).replace('HEAD', 'PHOT')
    with fits.open(phot_file_path) as photometry_hdulist:
        phot_data = Table(photometry_hdulist[1].data)

    # If using pandas instead of astropy on the above line
    # Avoid ValueError: Big-endian buffer not supported on little-endian compiler
    # by adding in the below code:
    # for key, val in phot_data.iteritems():
    #     phot_data[key] = phot_data[key].to_numpy().byteswap().newbyteorder()

    with tqdm(meta_data.iterrows(), total=len(meta_data), disable=not verbose) as pbar:
        for idx, meta in pbar:
            lc_start = int(meta['PTROBS_MIN']) - 1
            lc_end = int(meta['PTROBS_MAX'])
            lc = phot_data[lc_start: lc_end]
            lc.meta.update(meta)

            pbar.update()
            pbar.refresh()
            yield lc


def iter_lc_for_cadence_model(cadence: str, model: int, verbose: bool = True) -> Iterable[Table]:
    """Iterate over simulated light-curves  for a given cadence

    Args:
        cadence: Name of the cadence to summarize
        model: Model number to retrieve light-curves for
        verbose: Display a progress bar

    Yields:
        An Astropy table with the MJD and filter for each observation
    """

    total = count_light_curves(cadence, model)
    light_curve_iter = get_model_headers(cadence, model)

    with tqdm(light_curve_iter, desc=cadence, total=total, disable=not verbose) as pbar:
        for header_path in pbar:
            for lc in iter_lc_for_header(header_path, verbose=False):
                pbar.update()
                pbar.refresh()
                yield lc


def format_plasticc_sncosmo(light_curve: Table) -> Table:
    """Format a PLaSTICC light-curve to be compatible with sncosmo

    Args:
        light_curve: Table of PLaSTICC light-curve data

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
