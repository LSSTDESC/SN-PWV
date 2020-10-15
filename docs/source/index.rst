Impact of Chromatic Effects on LSST SNe
=======================================

Understanding the time variable behavior of the atmosphere is an important step
in calibrating data from ground-based astronomical surveys. This project explores
how chromatic effects induced by the atmosphere will impact observation taken
of Type Ia Supernova by the Legacy Survey of Space and Time (LSST).

API Usage
---------

This project is supported by a custom Python API that if free for public use
under the terms and conditions of the GNU General Public License (V 3.0, see
here). For questions concerning the Python API, please see the API section of
these docs or raise an issue on `GitHub <https://github.com/lsstdesc/sn-pwv>`_.
If your question is of a scientific nature, please also see the Project Notes
section.


Contributing
------------

All involvement with this project is subject to the policies of the Dark Energy Science
Collaboration, with particular emphasis on the **Code of Conduct** and
**Software Development Policy**. More information is available
`here <https://lsstdesc.org/pages/policies.html>`_.


.. toctree::
   :hidden:
   :maxdepth: 0
   :titlesonly:

   Overview<self>
   overview/install.rst
   overview/data_provenance.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Notebooks / Usage Examples

   notebooks/lsst_filters.nblink
   notebooks/pwv_eff_on_black_body.nblink
   notebooks/sne_delta_mag.nblink
   notebooks/simulating_lc_with_cadence.nblink

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference
   :titlesonly:

   api/sn_analysis.rst
   api/constants.rst
   api/filters.rst
   api/modeling.rst
   api/plasticc.rst
   api/plotting.rst
   api/reference.rst
   api/sn_magnitudes.rst
