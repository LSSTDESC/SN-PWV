#!/bin/bash
#SBATCH --job-name=snat_sim_core_allocation_scan
#SBATCH --output=core_allocation_scan.log
#
# Single CPU using Haswell 32 core machine
#SBATCH --qos=debug
#SBATCH -L SCRATCH,cfs
#SBATCH --constraint=haswell
#SBATCH --array=38,46,54,62


module purge
module load python

conda activate SN-PWV

python -m cProfile -o "core_allocation_$SLURM_ARRAY_TASK_ID.pstat" ../fitting_cli.py \
    --sim_pool_size 30 - $SLURM_ARRAY_TASK_ID \
    --fit_pool_size $SLURM_ARRAY_TASK_ID \
    --iter_lim 100 \
    --cadence alt_sched \
    --sim_variability epoch \
    --fit_variability seasonal \
    --out_path "core_allocation_$SLURM_ARRAY_TASK_ID.csv"
wait