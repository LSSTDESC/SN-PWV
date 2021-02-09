# Runs a subset of light-curves through the data analysis pipeline using different global configuration parameters
# Intended for use during pipeline verification/validation
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0     --fit_variability 0        -d ../../data/validation/pwv_sim_0_fit_0.h5            -o ../../data/validation/pwv_sim_0_fit_0.csv
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability 4     --fit_variability 4        -d ../../data/validation/pwv_sim_4_fit_4.h5            -o ../../data/validation/pwv_sim_4_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0     --fit_variability 4        -d ../../data/validation/pwv_sim_0_fit_4.h5            -o ../../data/validation/pwv_sim_0_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability 4        -d ../../data/validation/pwv_sim_epoch_fit_4.h5        -o ../../data/validation/pwv_sim_epoch_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability seasonal -d ../../data/validation/pwv_sim_epoch_fit_seasonal.h5 -o ../../data/validation/pwv_sim_epoch_fit_seasonal.csv
python ../fitting_cli.py -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability epoch    -d ../../data/validation/pwv_sim_epoch_fit_epoch.h5    -o ../../data/validation/pwv_sim_epoch_fit_epoch.csv
