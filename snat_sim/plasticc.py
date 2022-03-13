"""The ``plasticc`` module provides data access for PLAsTICC cadence
simulations stored on the local machine. Data is accessible by specifying
the cadence name and model number used in a given simulation.

Usage Example
-------------

The ``PLAsTICC`` class is responsible for handling data access:

.. doctest:: python

   >>> from snat_sim.plasticc import PLAsTICC

   >>> # Get a list of cadences available in the directory printed above
   >>> print(PLAsTICC.get_available_cadences())  #doctest:+SKIP
   >>> [ 'alt_sched' ]  #doctest:+SKIP

   >>> # Count the number of light-curves for a given cadence and SN model
   >>> lc_data = PLAsTICC('alt_sched', model=11)
   >>> num_lc = lc_data.count_light_curves()

The class provides **basic** data access via the construction of an iterator
over the observed cadence for each simulated light-curve. The iterator returns
the unique Id, parameters, and cadecne used in each simulation.

.. note:: You should expect the first evaluation of the iterator to be slow
   since it has to load data into memory as chunks.

.. code-block:: python

   >>> lc_iterator = lc_data.iter_cadence(iter_lim=10, verbose=False)
   >>> snid, sim_params, cadence = next(lc_iterator)

The light-curve simulated by PLAsTICC for each cadence can optionally
be included with the iterator:

.. code-block:: python

   >>> lc_iterator = lc_data.iter_cadence(iter_lim=10, include_lc=True, verbose=False)
   >>> snid, sim_params, cadence, lc = next(lc_iterator)

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

from . import types
from .data_paths import paths_at_init
from .models import LightCurve, ObservedCadence
from .types import NumericalParams

YieldedWithoutLC = Tuple[int, NumericalParams, ObservedCadence]
YieldedWithLC = Tuple[int, NumericalParams, ObservedCadence, LightCurve]


class PLAsTICC:
    """Data access object for PLAsTICC simulation data"""

    def __init__(self, cadence: str, model: int) -> None:
        """Data access object for PLAsTICC simulations performed using a given cadence and SN model

        Args:
            cadence: The cadence to load data for
            model: The numerical identifier of the PLAsTICC SN model used in the simulation
        """

        self.cadence = cadence
        self.model = model

    @staticmethod
    def get_available_cadences() -> List[str]:
        """Return a list of all cadences available in the working environment"""

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

    @overload
    @staticmethod
    def _iter_cadence_for_header(header_path: types.PathLike, include_lc: bool = False) -> Iterator[YieldedWithoutLC]:
        ...  # pragma: no cover

    @overload
    @staticmethod
    def _iter_cadence_for_header(header_path: types.PathLike, include_lc: bool = True) -> Iterator[YieldedWithLC]:
        ...  # pragma: no cover

    @staticmethod
    def _iter_cadence_for_header(header_path, include_lc=False):
        """Iterate over cadence data from a given header file

        Files are expected to be written in pairs of a header file 
        (`*HEAD.fits`) that stores target meta data and a photometry file 
        (`*PHOT.fits`) with simulated light-curves.

        Args:
            header_path: Path of the header file
            include_lc: Include the PLAsTICC simulated light-curve with iterator outputs

        Yields:
            - The supernova identifier (SNID)
            - The parameters used in the simulation
            - The cadence of the simulation
        """

        # Load meta data from the header file
        with fits.open(header_path) as header_hdulist:
            meta_data = pd.DataFrame(header_hdulist[1].data)

        # Load light-curves from the photometry file, This is slow
        phot_file_path = str(header_path).replace('HEAD', 'PHOT')
        with fits.open(phot_file_path) as photometry_hdulist:
            phot_data = Table(photometry_hdulist[1].data)

        # If using pandas instead of astropy on the above line you need to avoid
        #   ValueError: Big-endian buffer not supported on little-endian compiler
        # by adding in the below code:
        # for key, val in phot_data.iteritems():
        #     phot_data[key] = phot_data[key].to_numpy().byteswap().newbyteorder()

        for idx, meta in meta_data.iterrows():
            # Select the individual light-curve by it's indices
            lc_start = int(meta['PTROBS_MIN']) - 1
            lc_end = int(meta['PTROBS_MAX'])
            lc_data = phot_data[lc_start: lc_end]

            params = {
                'ra': meta['RA'],
                'dec': meta['DECL'],
                't0': meta['SIM_PEAKMJD'],
                'x1': meta['SIM_SALT2x1'],
                'c': meta['SIM_SALT2c'],
                'z': meta['SIM_REDSHIFT_CMB'],
                'x0': meta['SIM_SALT2x0']
            }

            times = lc_data['MJD']
            bands = ['lsst_hardware_' + f.lower().strip() for f in lc_data['FLT']]
            zero_point = lc_data['ZEROPT']
            cadence = ObservedCadence(
                obs_times=times,
                bands=bands,
                zp=zero_point,
                zpsys='AB',
                gain=1,
                skynoise=lc_data['SKY_SIG']
            )

            if include_lc:
                lc = LightCurve(
                    time=times,
                    band=bands,
                    flux=lc_data['FLUXCAL'],
                    fluxerr=lc_data['FLUXCALERR'],
                    zp=zero_point,
                    zpsys='AB',
                    phot_flag=lc_data['PHOTFLAG']
                )

                yield int(meta['SNID'].strip()), params, cadence, lc

            else:
                yield int(meta['SNID'].strip()), params, cadence

    @overload
    def iter_cadence(
            self, iter_lim: int = None, include_lc: bool = False, verbose: bool = True
    ) -> Iterator[YieldedWithoutLC]:
        ...  # pragma: no cover

    @overload
    def iter_cadence(
            self, iter_lim: int = None, include_lc: bool = True, verbose: bool = True
    ) -> Iterator[YieldedWithLC]:
        ...  # pragma: no cover

    def iter_cadence(self, iter_lim=None, include_lc=False, verbose=True):
        """Iterate over available cadence data for each supernova

        Args:
            iter_lim: Limit the number of iterated light-curves
            include_lc: Include the PLAsTICC simulated light-curve with iterator outputs
            verbose: Display a progress bar

        Yields:
            - The supernova identifier (SNID)
            - The parameters used in the simulation
            - The cadence of the simulation
        """

        max_lc = self.count_light_curves()
        total = min(iter_lim, max_lc) if iter_lim else max_lc

        i = 0
        with tqdm(self.get_model_headers(), desc=self.cadence, total=total, disable=not verbose) as pbar:
            for header_path in pbar:
                for chunk in self._iter_cadence_for_header(header_path, include_lc=include_lc):
                    pbar.update()
                    pbar.refresh()
                    yield chunk

                    i += 1
                    if i >= total:
                        return
