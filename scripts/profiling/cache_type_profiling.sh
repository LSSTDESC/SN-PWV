#!/bin/bash

# This batch job profiles the performance of the analysis pipeline using
# different kinds of memoization / caching routines

#SBATCH --job-name=snat_sim_cache_type_profiling
#SBATCH --output=cache_type_profiling.log
#SBATCH --qos=debug
#SBATCH -L SCRATCH,cfs
#SBATCH --constraint=haswell
#SBATCH --time=30:00
#SBATCH --array=500,1000,2000

# Set the cadence simulation and number of light-curves we want to use
export USE_CADENCE=alt_sched
export SIM_PROCESSES=10
export FITTING_PROCESSES=52
export SIM_VARIABILITY=epoch
export FIT_VARIABILITY=seasonal

# Instantiate python environment, including various environmental variables
module purge
module load python
conda activate SN-PWV

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2

export OUTPUT_DIR="cache_type_profiling/pwv_${FIT_VARIABILITY}/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p $OUTPUT_DIR

# Point the analysis code at the new data
export CADENCE_SIMS=$SCRATCH/Cadence

# Cache data is stored to a temporary directory
export SNAT_SIM_CACHE=$(mktemp -d -p $SCRATCH)

# Profile the sequentially run pipeline using no caching
export SNAT_SIM_CACHE_TYPE=0
python -m cProfile -o "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.pstat" ../fitting_cli.py \
  --sim_pool_size $SIM_PROCESSES \
  --fit_pool_size $FITTING_PROCESSES \
  --iter_lim $SLURM_ARRAY_TASK_ID \
  --cadence $USE_CADENCE \
  --sim_variability $SIM_VARIABILITY \
  --fit_variability $FIT_VARIABILITY \
  --out_path "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.csv"

# Profile the sequentially run pipeline using memoization caching
export SNAT_SIM_CACHE_TYPE=1
python -m cProfile -o "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.pstat" ../fitting_cli.py \
  --sim_pool_size $SIM_PROCESSES \
  --fit_pool_size $FITTING_PROCESSES \
  --iter_lim $SLURM_ARRAY_TASK_ID \
  --cadence $USE_CADENCE \
  --sim_variability $SIM_VARIABILITY \
  --fit_variability $FIT_VARIABILITY \
  --out_path "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.csv"

# Profile the sequentially run pipeline using joblib caching
# The first run profiles performance with no precalculated values available on disk
# The second run profiles performance when precalculated values are used from the previous run
export SNAT_SIM_CACHE_TYPE=2
python -m cProfile -o "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.pstat" ../fitting_cli.py \
  --sim_pool_size $SIM_PROCESSES \
  --fit_pool_size $FITTING_PROCESSES \
  --iter_lim $SLURM_ARRAY_TASK_ID \
  --cadence $USE_CADENCE \
  --sim_variability $SIM_VARIABILITY \
  --fit_variability $FIT_VARIABILITY \
  --out_path "$OUTPUT_DIR/cache_${SNAT_SIM_CACHE_TYPE}.csv"

python -m cProfile -o "$OUTPUT_DIR/cache_second_${SNAT_SIM_CACHE_TYPE}.pstat" ../fitting_cli.py \
  --sim_pool_size $SIM_PROCESSES \
  --fit_pool_size $FITTING_PROCESSES \
  --iter_lim $SLURM_ARRAY_TASK_ID \
  --cadence $USE_CADENCE \
  --sim_variability $SIM_VARIABILITY \
  --fit_variability $FIT_VARIABILITY \
  --out_path "$OUTPUT_DIR/cache_second_${SNAT_SIM_CACHE_TYPE}.csv"

rm -r $SNAT_SIM_CACHE
