Output Data Model
=================

Data from the analysis pipeline is written to disk in HDF5 (``.h5``) format.
The number of results written to a single ``.h5`` file is capped at 5,000.
If you are simulating a larger number of SNe, or have allocated multiple I/O
processes at runtime, then results will be written to multiple files.
The data model for each of these files is identical.

HDF5 Data Paths
---------------

Data is written to multiple tables within each HDF5 file.
The path of each table is provided below along with a summary of the corresponding data.

+-----------------------------+----------------------------------------------------------------------+
| Table Path                  | Description                                                          |
+=============================+======================================================================+
| ``message``                 | A human readable message describing the status of each simulation.   |
+-----------------------------+----------------------------------------------------------------------+
| ``simulation/params``       | The parameters used as model inputs when simulating each supernova.  |
+-----------------------------+----------------------------------------------------------------------+
| ``simulation/<ID>``         | The simulated photometry of a given supernova.                       |
+-----------------------------+----------------------------------------------------------------------+
| ``fitting/params``          | The parameters recovered from fitting a model to each supernova.     |
+-----------------------------+----------------------------------------------------------------------+
| ``fitting/covariance/<ID>`` | The covariance matrix determined when fitting a model to a given SN. |
+-----------------------------+----------------------------------------------------------------------+

message
^^^^^^^

The schema of the ``message``  table is listed below.
Every simulation has an entry in the message table regardless of whether
the simulation for a given SNe was successful.

+-----------------+---------------+---------------------------------------------------------------------+
| Column Name     | Data Type     | Description                                                         |
+=================+===============+=====================================================================+
| ``snid``        | String        | The ID of each simulated supernova.                                 |
+-----------------+---------------+---------------------------------------------------------------------+
| ``success``     | Bool          | Whether the simulation/fitting process executed successfully.       |
+-----------------+---------------+---------------------------------------------------------------------+
| ``message``     | String        | Status message describing the status of each simulation.            |
+-----------------+---------------+---------------------------------------------------------------------+

simulation/params
^^^^^^^^^^^^^^^^^

Simulation parameters are appended to the ``simulation/params`` for every single simulation
regardless of the success of the simulation.

+-----------------+---------------+---------------------------------------------------------------------+
| Column Name     | Data Type     | Description                                                         |
+=================+===============+=====================================================================+
| ``snid``        | String        | The ID of each simulated supernova.                                 |
+-----------------+---------------+---------------------------------------------------------------------+
| ``<param>``     | Float         | The value of the parameter used in the simulation.                  |
+-----------------+---------------+---------------------------------------------------------------------+

simulation/<ID>
^^^^^^^^^^^^^^^

The ``simulation/<ID>`` tables provide the simulated light curve for each supernova.

.. important:: The ``simulation/<ID>`` tables are only created if writing light-curves to
   disk is enabled via the appropriate command line arguments.

+-----------------+---------------+----------------------------------------------------------------------+
| Column Name     | Data Type     | Description                                                          |
+=================+===============+======================================================================+
| ``band``        | String        | The name of the bandpass for the simulated observation.              |
+-----------------+---------------+----------------------------------------------------------------------+
| ``flux``        | Float         | The simulated observed flux through a given band.                    |
+-----------------+---------------+----------------------------------------------------------------------+
| ``fluxerr``     | Float         | The error in ``flux``.                                               |
+-----------------+---------------+----------------------------------------------------------------------+
| ``zp``          | Float         | The photometric zero point of the observation.                       |
+-----------------+---------------+----------------------------------------------------------------------+
| ``zpsys``       | Float         | The name of the zero point system used when simulating flux.         |
+-----------------+---------------+----------------------------------------------------------------------+
| ``phot_flag``   | Integer       | Either 0 (non-detection), 4096 (detection), or 6144 (first trigger). |
+-----------------+---------------+----------------------------------------------------------------------+


fitting/params
^^^^^^^^^^^^^^

Entries in the ``fitting/params`` table are only provided for simulations that result in successful fits.
Any fits that fail to converge are not written to the table.

Precalculated B-band magnitudes are determined using the ``bessellb`` filter built-in to
sncosmo and the Betoule+ 2014 cosmology.

+-----------------------+---------------+---------------------------------------------------------------------+
| Column Name           | Data Type     | Description                                                         |
+=======================+===============+=====================================================================+
| ``snid``              | String        | The ID of each simulated supernova.                                 |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``fit_<param>``       | Float         | The value of the parameter recovered from the fit.                  |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``err_<param>``       | Float         | The estimated error in the fitted parameter.                        |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``chisq``             | Float         | The chisquared of the fit result.                                   |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``ndof``              | Float         | Number of degrees of freedom in the fit.                            |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``apparent_bessellb`` | Float         | Apparent b-band magnitude of the fitted model at peak.              |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``absolute_bessellb`` | Float         | Absolute b-band magnitude of the fitted model at peak.              |
+-----------------------+---------------+---------------------------------------------------------------------+

fitting/covariance/<ID>
^^^^^^^^^^^^^^^^^^^^^^^

Column names in the ``fitting/covariance/<ID>`` tables act as labels for the rows and columns of the covariance matrix.

+-----------------------+---------------+---------------------------------------------------------------------+
| Column Name           | Data Type     | Description                                                         |
+=======================+===============+=====================================================================+
| ``<param>``           | Float         | A column of the covariance matrix.                                  |
+-----------------------+---------------+---------------------------------------------------------------------+
