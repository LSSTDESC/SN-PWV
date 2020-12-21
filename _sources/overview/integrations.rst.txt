.. _integration_docs:

Package Integrations
====================

The ``snat_sim`` package is designed to automatically integrate with other Python
packages commonly used in scientific research. This integration extends the
functionality of various external packages and is performed automatically on
import. A description of how ``snat_sim`` integrates with different packages
is provided below. For a technical overview on the implementation of each
integration, see the `utils module documentation <../api/utils/utils.html>`_
or follow one of the links below.

Pandas
------

The ``pandas`` package is designed to support the manipulation of tabular data.
In addition to providing an impressive collection of built-in data-analysis tools,
the package also supports the implementation of
`custom accessors <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_
that can extend the capability of ``pandas`` objects.
Importing the ``snat_sim`` package will automatically register custom accessors
with the ``pandas`` package.

A summary of different groups of accessors is provided below, including
links to detailed documentation of the accessible functions.

Time Series Utilities
^^^^^^^^^^^^^^^^^^^^^

Time series utilities are available for `pandas.Series` objects and are accessible via the
``tsu`` accessor name.

.. py:currentmodule:: snat_sim.utils.time_series.TSUAccessor

.. autosummary::
   :nosignatures:

   periodic_interpolation
   resample_data_across_year
   supplemented_data

SNCosmo
-------

The ``sncosmo`` package is used to analyze spectra-photometric observations of supernovae.
The package includes a `registry system <https://sncosmo.readthedocs.io/en/latest/registry.html>`_
that allows users to retrieve supernova models, filter profiles, and other information by name.

By default, the ``sncosmo`` package comes pre-packaged with a number of filter response
curves from different astronomical surveys. The ``snat_sim`` package extends the number of
available filter profiles by registering additional filters for the Dark Energy Camera (DECam)
and the Legacy Survey of Space and Time (LSST). A summary of the registered filters is
provided below:

Dark Energy Camera (DECam)
^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------+-------------------------------------------------------------------------+
| Registered Filter Name | Filter Description                                                      |
+========================+=========================================================================+
| DECam_<ugrizY>_filter  | DECam optical response curves                                           |
+------------------------+-------------------------------------------------------------------------+
| DECam_atm              | Fiducial atmosphere assumed for the optical response curves             |
+------------------------+-------------------------------------------------------------------------+
| DECam_ccd              | DECam CCD Response curve                                                |
+------------------------+-------------------------------------------------------------------------+

Legacy Survey of Space and Time (LSST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------+-------------------------------------------------------------------------+
| Registered Filter Name | Filter Description                                                      |
+========================+=========================================================================+
| lsst_detector          |  Detector sensitivity defined in the LSST Science Requirements Document.|
+------------------------+-------------------------------------------------------------------------+
| lsst_atmos_10          |  Fiducial atmosphere over a 10 year baseline.                           |
+------------------------+-------------------------------------------------------------------------+
| lsst_atmos_std         |  Fiducial atmosphere likely for LSST at 1.2 airmasses.                  |
+------------------------+-------------------------------------------------------------------------+
| lsst_filter_<ugrizy>   |  Throughput of the <ugrizy> glass filters only.                         |
+------------------------+-------------------------------------------------------------------------+
| lsst_hardware_<ugrizy> |  Hardware contribution response curve in each band.                     |
+------------------------+-------------------------------------------------------------------------+
| lsst_total_<ugrizy>    |  Total response curve in each band.                                     |
+------------------------+-------------------------------------------------------------------------+
| lsst_m<123>            |  Response curve contribution from each mirror.                          |
+------------------------+-------------------------------------------------------------------------+
| lsst_lens<123>         |  Response curve contribution from each lens.                            |
+------------------------+-------------------------------------------------------------------------+
| lsst_mirrors           |  Combined result from all mirrors.                                      |
+------------------------+-------------------------------------------------------------------------+
| lsst_lenses            |  Combined response from all lenses.                                     |
+------------------------+-------------------------------------------------------------------------+
| lsst_<ugrizy>_no_atm   |  Throughput in each band without a fiducial atmosphere.                 |
+------------------------+-------------------------------------------------------------------------+

