Installation and Setup
======================

The ``snat_sim`` package is not available via a package manager, but can
be installed directly from the project's source code. Please follow the
steps outlined below to install and configure the package.

Running at NERSC
----------------

If you are running this analysis on the Cori filesystem at NERSC,
most of the setup procedure has already been completed for you.
The only manual setup task is to define an appropriate
Python environment for running the source code.

If you have not already done so, configure your Python interpreter as follows
(this only has to be done once).

.. code-block:: bash

   module load python
   conda init

Next, use the environment file included with the project source code to
define a new conda environment

.. code-block:: bash

   conda env create --file SN-PWV/cori_env.yml -y

You can now activate and deactivate the Python environment for this project at will.
When working inside the environment, the necessary environmental variables will already be set.
For example:

.. code-block:: bash

   conda activate SN-PWV
   echo $CADENCE_SIMS

Using a Custom Environment
--------------------------

For individuals developing on machines not linked to the CORI filesystem,
the project source code is publicly available and can be downloaded/configured
as outlined below.

Downloading the Source
----------------------

Source code for this project is available on
`GitHub <https://github.com/LSSTDESC/SN-PWV>`_ and can be downloaded directly
from the GitHub repository page or by using the ``git`` command line utility:

.. code-block:: bash

   git clone  --depth=1 --branch=master https://github.com/LSSTDESC/SN-PWV.git SN-PWV
   rm -rf ./SN-PWV/.git

The package can then be installed into your working environment manually, or
using ``pip``:

.. code-block:: bash

   cd SN-PWV
   pip install .

If you want to run the package's test suite, you will need to install
the test suite dependencies:

.. code-block:: bash

   pip install .[tests]

Extra dependencies are also required for building package documentation.
These requirements can be found in the ``docs`` directory, and installed as follows

.. code-block:: bash

   cd SN-PWV/docs/
   pip install .[docs]

Downloading Light-Curve Sims
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important:: Not all parts of this project require existing light-curve
   simulations. The complete data set takes up several hundred GB of storage.
   To avoid future headaches, please ensure you actually need this data and
   don't already have it available in your working environment.

Simulated light-curve data is hosted on `Zenodo <https://zenodo.org/>`_ and can be downloaded using
the ``wget`` command. URL's for various subsets of the data is listed are
listed in  ``file_list.txt`` and can be downloaded as shown below.
For convenience, the  ``timeout`` and ``tries`` arguments can be used to
indefinitely retry a failed download and the  ``continue`` flag can be
included to avoid restarting a failed download from scratch.

.. code-block:: bash

   wget --continue --timeout 0 --tries 0 -i data/plasticc/file_list.txt -P /desired/output/directory/

If you have difficulty downloading all the data at once, or if you don't
need the entire data set, try individually downloading the files listed
in ``file_list.txt`` . The downloaded files will be nested, compressed
files using a mix of the ``.gz`` and ``.tar.gz`` compression formats.
You can decompress them using the following commands:

.. code-block:: bash

   tar -xvzf [FILE TO DECOMPRESS].tar.gz
   gunzip [FILE TO DECOMPRESS]/*/*.gz

Configuring Your Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The path of the downloaded data needs to be specified in the project
environment so that the software knows where to find the simulated
light-curves.

.. code-block:: bash

   export CADENCE_SIMS="[DESIRED DATA DIRECTORY]"

If you are using a ``conda`` environment, this can be accomplished by
specifying the desired data directory as follows:

.. code-block:: bash

   # Instantiate the new environment
   conda activate [ENV-NAME]
   
   # Go to the environment's home directory
   cd $CONDA_PREFIX
   
   # Create files to run on startup and exit
   mkdir -p ./etc/conda/activate.d
   mkdir -p ./etc/conda/deactivate.d
   touch ./etc/conda/activate.d/env_vars.sh
   touch ./etc/conda/deactivate.d/env_vars.sh
   
   # Add environmental variables
   echo 'export CADENCE_SIMS="[DESIRED DATA DIRECTORY]"' >> ./etc/conda/activate.d/env_vars.sh
   echo 'unset CADENCE_SIMS' >> ./etc/conda/deactivate.d/env_vars.sh
   
   # Finally, don't forget to exit your environment
   conda deactivate
