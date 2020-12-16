Data Provenance
===============

This project takes advantage of published data from multiple external research
efforts. Listed below is a summary of
data sets used by this project and their origins.

Atmospheric Measurements
------------------------

**Repository Path:** Data access provided by the ``pwv_kpno`` `API <https://mwvgroup.github.io/pwv_kpno/>`_.

**Data Source:** `The SuomiNet Project <https://www.suominet.ucar.edu/>`_

**Description:**
    Meteorological measurements taken at various global positions are used to
    characterize atmospheric variability over time. This data is taken by
    the SuomiNet project and it's affiliated teams / projects. Published values
    include the pressure, temperature, relative humidity, and precipitable water
    vapor concentration (at zenith) sample at 30 minute intervals.

CTIO Filters
------------

**Repository Path:** *data/filters/ctio/**

**Data Source:** `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/theory/fps/>`_

**Description:**
    Filter response curves corresponding to the Dark Energy Camera (DECam) used
    by the Cerro Telolo International Observatory (CTIO). These filter profiles
    were used primarily as a validation step during early development, and were
    later replaced in the analysis by the LSST filter set.


LSST Filters
------------

**Repository Path:** *data/filters/lsst/**

**Data Source:** Cloned from https://github.com/lsst/throughputs

**Description:**
    Filter response curves curves considered as the 'baseline' performance of LSST.
    Throughput curves for other surveys are also available from the parent website,
    but only the LSST related data is used for this project. The filter profiles
    are generally identical to the profiles considered in the LSST Science
    Requirements Document (SRD).

Light-Curve Simulations
-----------------------

**Repository Path:** *Not included with project source code*

**Data Source:** Data is published by DESC and hosted on `Zenodo <https://zenodo.org/>`_.

**Description:**
    SN light-curves were simulated using a variety of cadences and supernova models.
    Developers can reference the assumed data model `here <plasticc_model.html>`_.

Stellar Spectra
---------------

**Repository Path:** *data/stellar_spectra/**

**Data Source:** Sourced from the `Goettingen Spectral Library <http://phoenix.astro.physik.uni-goettingen.de/?page_id=15>`_.

**Description:**
    The Goettingen Spectral Library provides high resolution stellar spectra representing
    a variety of stellar types. Spectra were downloaded for a handful of spectral types, with
    additional spectra being downloaded on an "as needed" basis.
