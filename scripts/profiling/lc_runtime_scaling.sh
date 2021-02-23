#!/bin/bash
#SBATCH --job-name=snat_sim_core_allocation_scan
#SBATCH --output=core_allocation_scan.log
#
# Single CPU using Haswell 32 core machine
#SBATCH --qos=debug
#SBATCH -L SCRATCH,cfs
#SBATCH --constraint=haswell
#SBATCH --array=4000,8000,12000,16000,20000
#SBATCH --time=30:00

module purge
module load python

conda activate SN-PWV

export NON_IO_PROCESSES=62 # 64 cores total minus two processes for I/O
export FITTING_PROCESSES=52

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2

python -m cProfile -o "lc_runtime_scaling_$SLURM_ARRAY_TASK_ID.pstat" ../fitting_cli.py \
  --sim_pool_size $(($NON_IO_PROCESSES - $SLURM_ARRAY_TASK_ID)) \
  --fit_pool_size $FITTING_PROCESSES \
  --iter_lim $SLURM_ARRAY_TASK_ID \
  --cadence alt_sched \
  --sim_variability epoch \
  --fit_variability seasonal \
  --out_path "core_allocation_$SLURM_ARRAY_TASK_ID.csv"
wait
