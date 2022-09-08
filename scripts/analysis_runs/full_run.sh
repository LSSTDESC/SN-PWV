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

OUT_DIR=$OUT_DIR/
mkdir -p $OUT_DIR

echo "Results generated using snat_sim version $(snat-sim --version)" > $OUT_DIR/version.txt
snat-sim          --bound_t0 $MIN_T0 $MAX_T0 --bound_x1 $MIN_STRETCH $MAX_STRETCH --bound_c $MIN_COLOR $MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability 4        -o $OUT_DIR/pwv_sim_epoch_fit_4/out.h5
snat-sim          --bound_t0 $MIN_T0 $MAX_T0 --bound_x1 $MIN_STRETCH $MAX_STRETCH --bound_c $MIN_COLOR $MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability seasonal -o $OUT_DIR/pwv_sim_epoch_fit_seasonal/out.h5
snat-sim -i 10000 --bound_t0 $MIN_T0 $MAX_T0 --bound_x1 $MIN_STRETCH $MAX_STRETCH --bound_c $MIN_COLOR $MAX_COLOR --write_lc_sims -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability epoch    -o $OUT_DIR/pwv_sim_epoch_fit_epoch/out.h5
