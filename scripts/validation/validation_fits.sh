python ../fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability 0     --fit_variability 0        -d pwv_sim_0_fit_0 -o pwv_sim_0_fit_0.csv
python ../fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability 4     --fit_variability 4        -d pwv_sim_4_fit_4 -o pwv_sim_4_fit_4.csv
python ../fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability 4        -d pwv_sim_epoch_fit_4 -o pwv_sim_epoch_fit_4.csv
python ../fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability seasonal -d pwv_sim_epoch_fit_seasonal -o pwv_sim_epoch_fit_seasonal.csv
python ../fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability epoch    -d pwv_sim_epoch_fit_epoch -o pwv_sim_epoch_fit_epoch.csv
