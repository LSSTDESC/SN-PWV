# Full pipeline runs over an entire cadence

NUM_SIM=53
NUM_FIT=8
NUM_WRITE=2

python ../../snat_sim_cli.py --overwrite -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability 4        -o ../../data/analysis_runs/pwv_sim_epoch_fit_4/out.h5
python ../../snat_sim_cli.py --overwrite -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability seasonal -o ../../data/analysis_runs/pwv_sim_epoch_fit_seasonal/out.h5
python ../../snat_sim_cli.py --iterlim 100000 --overwrite -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE -c alt_sched_rolling --sim_variability epoch --fit_variability epoch -o ../../data/analysis_runs/pwv_sim_epoch_fit_seasonal/out.h5
