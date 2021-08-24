"""The ``utils`` module is responsible for handling the integration
of external packages with ``snat_sim``. It is also responsible for
automatically configuring the host python environment during package
import.

.. important:: Unless you are looking to extend the underlying functionality or
   behavior of how ``snat_sim`` integrates with third party packages, you
   most likely aren't interested in importing directly from this module.
   See the :ref:`integration_docs` section of the docs for more information.

SubModules
----------

.. autosummary::
   :nosignatures:

   caching
   cov_utils
   filters
   time_series

Module Docs
-----------
"""

from . import *


def setup_environment() -> None:
    """Register package integrations with third party dependencies"""

    # Importing these modules auto registers them with pandas
    from . import cov_utils
    from . import time_series

    # Register custom filter profiles with sncosmo
    from .filters import register_lsst_filters, register_decam_filters
    register_lsst_filters()
    register_decam_filters()
