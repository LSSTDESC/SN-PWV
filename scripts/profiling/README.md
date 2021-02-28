This directory contains a series of script for profiling the `snat_sim` analysis 
pipeline and visualizing the results. Profiling jobs should be submitted using SLURM
and the included shell scripts. Profiling results can can be visualized using the
included Python scripts.  

## Setup

Run the following setup tasks in login node before job submission
1. Enable `conda` from the login node. This only has to be run once.

```bash
module purge
module load python
conda init
```

2. Make sure the ``SN-PWV`` conda environment is defined as follows.

```bash
wget https://raw.githubusercontent.com/LSSTDESC/SN-PWV/master/cori_env.yml
conda env create --file cori_env.yml
```

3. Copy profiling data to scratch directory if not already available. The `alt_sched` cadence is used by default.

```bash
mkdir -p $SCRATCH/Cadence/alt_sched
cp -u -r $CADENCE_SIMS/alt_sched/LSST_WFD_alt_sched_MODEL11 $SCRATCH/Cadence/alt_sched/
```