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

+-----------------+---------------+---------------------------------------------------------------------+
| Column Name     | Data Type     | Description                                                         |
+=================+===============+=====================================================================+
| ``snid``        | String        | The ID of each simulated supernova.                                 |
+-----------------+---------------+---------------------------------------------------------------------+
| ``<param>``     | Float         |                                                                     |
+-----------------+---------------+---------------------------------------------------------------------+

simulation/<ID>
^^^^^^^^^^^^^^^

fitting/params
^^^^^^^^^^^^^^

+-----------------------+---------------+---------------------------------------------------------------------+
| Column Name           | Data Type     | Description                                                         |
+=======================+===============+=====================================================================+
| ``snid``              | String        | The ID of each simulated supernova.                                 |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``fit_<param>``       | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``err_<param>``       | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``chisq``             | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``ndof``              | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``apparent_bessellb`` | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+
| ``absolute_bessellb`` | Float         |                                                                     |
+-----------------------+---------------+---------------------------------------------------------------------+

fitting/covariance/<ID>
^^^^^^^^^^^^^^^^^^^^^^^


