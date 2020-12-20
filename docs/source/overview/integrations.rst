External Integrations
=====================

The ``snat_sim`` package is designed to automatically integrate with other Python
packages commonly used in scientific research. This integration extends the
functionality of various external packages and is performed automatically on
import. A description of how ``snat_sim`` integrates with different packages
is provided below. For a complete technical overview, see the
`utils module documentation <../api/utils/utils.html>`_.

Pandas
------

The ``pandas`` package is designed to support the manipulation of tabular data.
In addition to providing an impressive collection of built-in data-analysis tools,
the package also supports the implementation of
`custom accessors <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_
that can extend the capability of ``pandas`` objects.
Importing the ``snat_sim`` package will automatically register custom accessors
with the ``pandas`` package.

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


