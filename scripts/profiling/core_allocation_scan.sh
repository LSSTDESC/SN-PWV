#!/bin/bash
#SBATCH --job-name=snat_sim_core_allocation_scan
#SBATCH --output=core_allocation_scan.log
#
# Single CPU using Haswell 32 core machine
#SBATCH --qos=debug
#SBATCH -L SCRATCH,cfs
#SBATCH --constraint=haswell
#SBATCH --array=42,46,50,54,58

module purge
module load python
conda activate SN-PWV

# 64 cores total minus two processes for I/O
export NON_IO_PROCESSES=62

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2

python -m cProfile -o "core_allocation_$SLURM_ARRAY_TASK_ID.pstat" ../fitting_cli.py \
  --sim_pool_size $(($NON_IO_PROCESSES - $SLURM_ARRAY_TASK_ID)) \
  --fit_pool_size $SLURM_ARRAY_TASK_ID \
  --iter_lim 250 \
  --cadence alt_sched \
  --sim_variability epoch \
  --fit_variability seasonal \
  --out_path "core_allocation_$SLURM_ARRAY_TASK_ID.csv"
wait
