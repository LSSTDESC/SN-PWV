#!/bin/bash
#SBATCH --job-name=snat_sim_cache_type_profiling
#SBATCH --output=cache_type_profiling.log
#SBATCH --qos=debug
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32000
#SBATCH --licenses=cscratch1
#SBATCH --constraint=haswell
#SBATCH --time=60:00

module purge
module load python

conda activate SN-PWV

export SNAT_SIM_CACHE_TYPE=0
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py alt_sched_small

export SNAT_SIM_CACHE_TYPE=1
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py alt_sched_small

export SNAT_SIM_CACHE_TYPE=2
python -m cProfile -o "cache_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py alt_sched_small
python -m cProfile -o "cache_second_$SNAT_SIM_CACHE_TYPE.pstat" sequential_profiling.py alt_sched_small
wait
