"""The ``data_paths`` module is responsible for pointing at local file paths
by the parent package.

The ``DataPaths`` provides a single access point for retrieving all external
file paths used by the parent package. Default paths are assumed automatically,
unless customized options are defined in the working environment (see the docs
for more details.)


Usage Example
-------------

To identify the current paths as defined in the working environemt:

.. doctest::

   >>> from snat_sim.data_paths import DataPaths
   >>>
   >>> current_paths = DataPaths()
   >>> spectra_dir = current_paths.stellar_spectra_dir
   >>> print(type(spectra_dir))
   <class 'pathlib.PosixPath'>

A record is kept of any custom paths defined in the working environment at the
time of package instantiation. These paths are available as an instance of the
``DataPaths`` class:


   >>> from snat_sim.data_paths import paths_at_init
   >>>
   >>> print(type(paths_at_init))
   <class 'snat_sim.data_paths.DataPaths'>

Module Docs
-----------
"""

import os
from pathlib import Path
from typing import Optional


class DataPaths:
    """Points to the location of on disk data used by the parent package"""

    def __init__(self) -> None:
        """Establish the location of package data on the local machine"""

        self.data_dir = Path(
            os.environ.get(
                'SNAT_SIM_DATA',
                Path(__file__).resolve().parent.parent / 'data'
            )
        ).resolve()

        try:
            self._plasticc_directory = Path(os.environ['CADENCE_SIMS']).resolve()

        except KeyError:
            self._plasticc_directory = self.data_dir / 'plasticc'

    @property
    def joblib_path(self) -> Path:
        """Directory to store function calls cached by joblib"""

        return self.data_dir / 'joblib'

    @property
    def _config_path(self) -> Path:
        """The path of the PWV configuration file"""

        return self.data_dir / 'defaults' / 'ref_pwv.yaml'

    @property
    def stellar_spectra_dir(self) -> Path:
        """Directory with stellar spectra"""

        return self.data_dir / 'stellar_spectra'

    @property
    def stellar_flux_dir(self) -> Path:
        """Directory with stellar flux values"""

        return self.data_dir / 'stellar_fluxes'

    def get_filters_dir(self, survey: str = None) -> Path:
        """Directory with filter profiles

        Args:
            survey: Return subdirectory for the given filter
        """

        path = self.data_dir / 'filters'
        if survey:
            path /= survey

        return path

    def get_plasticc_dir(self, cadence: Optional[str] = None, model: Optional[int] = None) -> Path:
        """Directory with PLaSTICC simulation data

        Args:
            cadence: Return subdirectory for the given simulation cadence
            model: Return subdirectory for the given simulation model
        """

        plasticc_directory = self._plasticc_directory
        if cadence:
            plasticc_directory /= cadence

            if model:
                plasticc_directory /= f'LSST_WFD_{cadence}_MODEL{model}'

        elif model:
            raise ValueError('``model`` cannot be defined without also specifying the ``cadence``.')

        return plasticc_directory


# Prebuilt object representing the ``data_dir`` as determined at package init
# Objects instantiated later on may have a different ``data_dir`` if the variables
# in the working environment change during runtime
paths_at_init = DataPaths()
