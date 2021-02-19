#!/bin/bash
# ------------------------ Setup / Documentation -----------------------
# This script profiles the performance of the analysis pipeline using
# different kinds of memoization / caching routines
#
# Run the following setup tasks in login node before job submission
# Copy profiling data to scratch directory if not already available
#    mkdir -p $SCRATCH/Cadence/alt_sched
#    cp -u -r $CADENCE_SIMS/alt_sched/LSST_WFD_alt_sched_MODEL11 $SCRATCH/Cadence/alt_sched/
# ----------------------------------------------------------------------

#SBATCH --job-name=snat_sim_cache_type_profiling
#SBATCH --output=cache_type_profiling.log
#SBATCH --qos=debug
#SBATCH -L SCRATCH,cfs
#SBATCH --constraint=haswell
#SBATCH --time=30:00

# Set the cadence simulation and number of light-curves we want to use
export USE_CADENCE=alt_sched
export NUMBER_SIMS=10

# Instantiate python environment, including various environmental variables
module purge
module load python
conda activate SN-PWV

# Point the analysis code at the new data
export CADENCE_SIMS=$SCRATCH/Cadence

# Cache data is stored to a temporary directory
export SNAT_SIM_CACHE=$(mktemp -d -p $SCRATCH)

# Profile the sequentially run pipeline using no caching
export SNAT_SIM_CACHE_TYPE=0
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE $NUMBER_SIMS

# Profile the sequentially run pipeline using memoization caching
export SNAT_SIM_CACHE_TYPE=1
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE $NUMBER_SIMS

# Profile the sequentially run pipeline using joblib caching
# The first run profiles performance with no precalculated values available on disk
# The second run profiles performance when precalculated values are used from the previous run
export SNAT_SIM_CACHE_TYPE=2
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE $NUMBER_SIMS
python -m cProfile -o "cache_second_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE $NUMBER_SIMS

rm -r $SNAT_SIM_CACHE
