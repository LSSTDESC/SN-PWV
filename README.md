# PWV Effects on LSST SNe

[![Build Status](https://www.travis-ci.com/LSSTDESC/SN-PWV.svg?branch=master)](https://www.travis-ci.com/LSSTDESC/SN-PWV)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/SN-PWV/badge.svg?branch=configure_coverage)](https://coveralls.io/github/LSSTDESC/SN-PWV?branch=configure_coverage)

Investigating the impact of chromatic effects on LSST SNe.

## Setup

#### Step 1: Install Dependencies

Project dependencies can be installed with pip:

```python
pip install -r requirements.txt
```

#### Step 2: Download Light-Curve Simulations

Please note that the downloaded data will take up a several hundred GB of storage. To avoid future headaches, choose where you will save the downloaded data with care. 



The simulated light-curves can be downloaded automatically using `wget` and the `file_list.txt` file from this repository. For convenience, the  `timeout` and `tries` arguments can be used to indefinitely retry a failed download and the  `continue` flag can be included to avoid restarting a failed download from scratch.

```bash
wget --continue --timeout 0 --tries 0 -o wget_log -i data/plasticc/file_list.txt -P /desired/output/directory/
```



The downloaded files will be nested, compressed files using a mix of the `.gz` and `.tar.gz` formats. You can decompress them using the following commands:

```bash
tar -xvzf file_to_decompress.tar.gz --verbose
gunzip file_to_decompress/*/*.gz --verbose
```



#### Step 3: Specify Data Location in Environment

The path of the downloaded data needs to be specified in the project environment so that the software knows where to find the simulated light-curves. If you are using a `conda` environment, this can be accomplished by replacing `/desired/output/directory/` in the below script:

```bash
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
```


