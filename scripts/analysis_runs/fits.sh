# Execute the analysis pipeline for a full analysis

# The number of processes to allocate for each stage in the pipeline
# See profiling results in the data directory when choosing these numbers.
NUM_SIM=53
NUM_FIT=8
NUM_WRITE=2

# Boundaries on fitted parameters
# t0 limits are defined relative to the true value
MIN_T0=-2
MAX_T0=2

# Stretch (x1) and color (c) limits are defined relative to 0
MIN_STRETCH=-3.5
MAX_STRETCH=2.5

MIN_COLOR=-0.3
MAX_COLOR=0.4

# Full analysis runs
python ../../snat_sim_cli.py -i 10000 --bound_t0 MIN_T0 MAX_T0 --bound_x1 MIN_STRETCH MAX_STRETCH --bound_c MIN_COLOR MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability epoch    -o ../../data/analysis_runs/pwv_sim_epoch_fit_epoch/out.h5
python ../../snat_sim_cli.py          --bound_t0 MIN_T0 MAX_T0 --bound_x1 MIN_STRETCH MAX_STRETCH --bound_c MIN_COLOR MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability 4        -o ../../data/analysis_runs/pwv_sim_epoch_fit_4/out.h5
python ../../snat_sim_cli.py          --bound_t0 MIN_T0 MAX_T0 --bound_x1 MIN_STRETCH MAX_STRETCH --bound_c MIN_COLOR MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability seasonal -o ../../data/analysis_runs/pwv_sim_epoch_fit_seasonal/out.h5
