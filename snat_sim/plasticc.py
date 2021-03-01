"""The ``plasticc`` module provides data access for PLaSTICC light-curve
simulations stored on the local machine. Data is accessible by specifying
the cadence and model used in a given simulation.

Usage Example
-------------

The ``PLaSTICC`` class is responsible to handling data access for
simulated light-curve data:

.. doctest:: python

   >>> from snat_sim.plasticc import PLaSTICC

   >>> # Get a list of cadences available in the directory printed above
   >>> print(PLaSTICC.get_available_cadences())  #doctest:+SKIP
   >>> [ 'alt_sched' ]  #doctest:+SKIP

   >>> # Count the number of light-curves for a given cadence and SN model
   >>> lc_data = PLaSTICC('alt_sched', model=11)
   >>> num_lc = lc_data.count_light_curves()

Te class provides **basic** data access via the construction of an iterator
over all available light-curves for the given cadence / model. You should expect
the first evaluation of the iterator to be slow since it has to load
light-curve data into memory as chunks.

.. code-block:: python

   >>> lc_iterator = lc_data.iter_lc(iter_lim=10, verbose=False)
   >>> plasticc_lc = next(lc_iterator)

PLaSTICC simulations were run using the ``SNANA`` package in FORTRAN and thus
are returned using the ``SNANA`` data model. Alternatively, you can convert the
returned tables to the data model used by the ``sncosmo`` Python package.

.. code-block:: python

   >>> formatted_lc = PLaSTICC.format_data_to_sncosmo(plasticc_lc)

Module Docs
-----------
"""

from pathlib import Path
from typing import *
from typing import List

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from .data_paths import paths_at_init


class PLaSTICC:
    """Data access object for PLaSTICC simulation data"""

    def __init__(self, cadence: str, model: int) -> None:
        """Data access object for PLaSTICC light-curve simulations performed using a given cadence and SN model

        Args:
            cadence: The cadence to load simulations for
            model: The numerical identifier of the PLaSTICC SN model used in the simulation
        """

        self.cadence = cadence
        self.model = model

    @staticmethod
    def format_data_to_sncosmo(light_curve: Table) -> Table:
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

    @staticmethod
    def get_available_cadences() -> List[str]:
        """Return a list of all available cadences available in the working environment"""

        return [p.name for p in paths_at_init.get_plasticc_dir().glob('*') if p.is_dir()]

    def get_model_headers(self) -> List[Path]:
        """Return a list of file paths for all simulation header files"""

        return list(paths_at_init.get_plasticc_dir(self.cadence, self.model).glob('*HEAD.FITS'))

    def count_light_curves(self) -> int:
        """Return the number of available light-curve simulations for the current cadence and model"""

        total_lc = 0
        for header_path in self.get_model_headers():
            with fits.open(header_path) as _temp:
                total_lc += len(_temp[1].data)

        return total_lc

    @staticmethod
    def _iter_lc_for_header(header_path: Union[Path, str], verbose: bool = True):
        """Iterate over light-curves from a given header file

        Files are expected to be written in pairs of a header file 
        (`*HEAD.fits`) that stores target meta data and a photometry file 
        (`*PHOT.fits`) with simulated light-curves.

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
                # Select the individual light-curve by it's indices
                lc_start = int(meta['PTROBS_MIN']) - 1
                lc_end = int(meta['PTROBS_MAX'])
                lc = phot_data[lc_start: lc_end]
                lc.meta.update(meta)

                pbar.update()
                pbar.refresh()

                yield lc

    def iter_lc(self, iter_lim: int = None, verbose: bool = True) -> Iterable[Table]:
        """Iterate over simulated light-curves  for a given cadence

        Args:
            iter_lim: Limit the number of iterated light-curves
            verbose: Display a progress bar

        Yields:
            An Astropy table with simulated light-curve data
        """

        max_lc = self.count_light_curves()
        total = min(iter_lim, max_lc) if iter_lim else max_lc

        i = 0
        with tqdm(self.get_model_headers(), desc=self.cadence, total=total, disable=not verbose) as pbar:
            for header_path in pbar:
                for lc in self._iter_lc_for_header(header_path, verbose=False):
                    pbar.update()
                    pbar.refresh()
                    yield lc

                    i += 1
                    if i >= total:
                        return
