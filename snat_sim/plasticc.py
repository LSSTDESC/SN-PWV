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

import os
from pathlib import Path
from warnings import warn

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from . import constants as const, lc_simulation

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'plasticc'


def get_data_dir():
    """Return the directory where the package expects PLaSTICC simulation to be located

    This value is the same as the environmental ``CADENCE_SIMS`` directory.
    If the environmental variable is not set, defaults to the project's
    ``data`` directory.

    Args:
        A ``Path`` object pointing to the data directory
    """

    try:
        plasticc_simulations_directory = Path(os.environ['CADENCE_SIMS'])

    except KeyError:
        warn(f'``CADENCE_SIMS`` is not set in environment. Defaulting to {DEFAULT_DATA_DIR}')
        plasticc_simulations_directory = DEFAULT_DATA_DIR

    return plasticc_simulations_directory


def get_available_cadences():
    """Return a list of all available cadences in the PLaSTICC simulation directory"""

    return [p.name for p in get_data_dir().glob('*') if p.is_dir()]


def get_model_headers(cadence, model):
    """Return a list of all header files for a given cadence and model

    Default is model 11 (Normal SNe)

    Args:
        cadence (str): Name of the cadence to list header files for
        model   (int): Model number to retrieve header paths for

    Returns:
        A list of Path objects
    """

    sim_dir = get_data_dir() / cadence / f'LSST_WFD_{cadence}_MODEL{model}'
    return list(sim_dir.glob('*HEAD.FITS'))


def count_light_curves(cadence, model):
    """Return the number of available light-curve simulations for a given cadence and model

    Args:
        cadence (str): Name of the cadence to list header files for
        model   (int): Model number to retrieve header paths for

    Returns:
        Number of simulated light-curves available in the working environment
    """

    cadence_header_files = get_model_headers(cadence, model)

    total_lc = 0
    for header_path in cadence_header_files:
        with fits.open(header_path) as _temp:
            total_lc += len(_temp[1].data)

    return total_lc


def iter_lc_for_header(header_path, verbose=True):
    """Iterate over light-curves from a given header file

    Files are expected in pairs of a header file (`*HEAD.fits`) that stores target
    meta data and a photometry file (`*PHOT.fits`) with simulated light-curves.

    Args:
        header_path     (Path, str): Path of the header file
        verbose (bool): Display a progress bar

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

            yield lc
            pbar.update(1)
            pbar.refresh()


def iter_lc_for_cadence_model(cadence, model, verbose=True):
    """Iterate over simulated light-curves  for a given cadence

    Args:
        cadence  (str): Name of the cadence to summarize
        model    (int): Model number to retrieve light-curves for
        verbose (bool): Display a progress bar

    Yields:
        An Astropy table with the MJD and filter for each observation
    """

    total = count_light_curves(cadence, model)
    light_curve_iter = get_model_headers(cadence, model)

    with tqdm(light_curve_iter, desc=cadence, total=total, disable=not verbose) as pbar:
        for header_path in pbar:
            for lc in iter_lc_for_header(header_path, verbose=False):
                yield lc
                pbar.update(1)
                pbar.refresh()


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


def extract_cadence_data(light_curve, zp=25, gain=1, skynoise=0, drop_nondetection=False):
    """Extract the observational cadence from a PLaSTICC light-curve

    Returned table is formatted for use with ``sncosmo.realize_lcs``.

    Args:
        light_curve      (Table): Astropy table with PLaSTICC light-curve data
        zp        (float, array): Overwrite the PLaSTICC zero-point with this value
        gain             (float): Gain to use during simulation
        skynoise    (int, array): Simulated skynoise in counts
        drop_nondetection (bool): Drop data with PHOTFLAG == 0

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
    observations['skynoise'] = skynoise
    return observations


def duplicate_plasticc_sncosmo(
        light_curve, model, zp=None, gain=1, skynoise=None, scatter=True, cosmo=const.betoule_cosmo):
    """Simulate a light-curve with sncosmo that matches the cadence of a PLaSTICC light-curve

    Args:
        light_curve  (Table): Astropy table with PLaSTICC light-curve data
        model      (SNModel): SNModel to use when simulating light-curve flux
        zp    (float, array): Optionally overwrite the PLaSTICC zero-point with this value
        gain         (float): Gain to use during simulation
        skynoise     (float):  Optionally overwrite the PLaSTICC skynoise with this value
        scatter       (bool): Add random noise to the flux values
        cosmo    (Cosmology): Rescale the ``x0`` parameter according to the given cosmology

    Returns:
        Astropy table with data for the simulated light-curve
    """

    use_redshift = 'SIM_REDSHIFT_CMB'
    if cosmo is None:
        x0 = light_curve.meta['SIM_SALT2x0']

    else:
        x0 = lc_simulation.calc_x0_for_z(light_curve.meta[use_redshift], 'salt2', cosmo=cosmo)

    # Params double as simulation parameters and meta-data meta data
    params = {
        'SNID': light_curve.meta['SNID'],
        'ra': light_curve.meta['RA'],
        'dec': light_curve.meta['DECL'],
        't0': light_curve.meta['SIM_PEAKMJD'],
        'x1': light_curve.meta['SIM_SALT2x1'],
        'c': light_curve.meta['SIM_SALT2c'],
        'z': light_curve.meta[use_redshift],
        'x0': x0
    }

    # Simulate the light-curve
    zp = zp if zp is not None else light_curve['ZEROPT']
    skynoise = skynoise if skynoise is not None else light_curve['SKY_SIG']
    observations = extract_cadence_data(light_curve, zp=zp, gain=gain, skynoise=skynoise)
    return lc_simulation.simulate_lc(observations, model, params, scatter=scatter)
