#!/bin/bash
#SBATCH --job-name=snat_sim_cache_type_profiling
#SBATCH --output=cache_type_profiling.log
#SBATCH --qos=debug
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32000
#SBATCH --licenses=cscratch1
#SBATCH --constraint=haswell
#SBATCH --time=60:00

# Global configuration options
# Set the cadence simulation we want to use
export USE_CADENCE=alt_sched_small

# Instantiate python environment, including various environmental variables
module purge
module load python
conda activate SN-PWV

# Copy profiling data to scratch directory if not already available
mkdir $SCRATCH/Cadence
cp -u -r $CADENCE_SIMS/$USE_CADENCE $SCRATCH/Cadence

# Point the analysis code at the new data
# Cache data is stored to a temporary directory
export CADENCE_SIMS=$SCRATCH/Cadence
export SNAT_SIM_CACHE=$(mktemp -d -p $SCRATCH)

# Profile the sequentially run pipeline using no caching
export SNAT_SIM_CACHE_TYPE=0
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE

# Profile the sequentially run pipeline using memoization caching
export SNAT_SIM_CACHE_TYPE=1
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE

# Profile the sequentially run pipeline using joblib caching
# The first run profiles performance with no precalculated values available on disk
# The second run profiles performance when precalculated values are used from the previous run
export SNAT_SIM_CACHE_TYPE=2
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE
python -m cProfile -o "cache_second_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py $USE_CADENCE

rm -r $SNAT_SIM_CACHE
wait
