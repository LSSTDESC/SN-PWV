export CADENCE_SIMS='/mnt/md0/sn-sims/'
python ../fitting_cli.py -s 1 -f 10 -i 500  -c alt_sched --sim_variability 0     --fit_variability 0        -d ../../data/validation/pwv_sim_0_fit_0            -o ../../data/validation/pwv_sim_0_fit_0.csv
python ../fitting_cli.py -s 1 -f 10 -i 500  -c alt_sched --sim_variability 4     --fit_variability 4        -d ../../data/validation/pwv_sim_4_fit_4            -o ../../data/validation/pwv_sim_4_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 500  -c alt_sched --sim_variability 4     --fit_variability 0        -d ../../data/validation/pwv_sim_4_fit_0            -o ../../data/validation/pwv_sim_4_fit_0.csv
python ../fitting_cli.py -s 1 -f 10 -i 500  -c alt_sched --sim_variability 0     --fit_variability 4        -d ../../data/validation/pwv_sim_0_fit_4            -o ../../data/validation/pwv_sim_0_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 1500 -c alt_sched --sim_variability epoch --fit_variability 4        -d ../../data/validation/pwv_sim_epoch_fit_4        -o ../../data/validation/pwv_sim_epoch_fit_4.csv
python ../fitting_cli.py -s 1 -f 10 -i 1500 -c alt_sched --sim_variability epoch --fit_variability seasonal -d ../../data/validation/pwv_sim_epoch_fit_seasonal -o ../../data/validation/pwv_sim_epoch_fit_seasonal.csv
python ../fitting_cli.py -s 1 -f 10 -i 1500 -c alt_sched --sim_variability epoch --fit_variability epoch    -d ../../data/validation/pwv_sim_epoch_fit_epoch    -o ../../data/validation/pwv_sim_epoch_fit_epoch.csv
