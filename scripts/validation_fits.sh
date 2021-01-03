python fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability 0     --fit_variability 0 -o validation/pwv_sim_0_fit_0.csv
python fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability 4     --fit_variability 4 -o validation/pwv_sim_4_fit_4.csv
python fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability 4  -o validation/pwv_sim_epoch_fit_4.csv
python fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability seasonal -o validation/pwv_sim_epoch_fit_seasonal.csv
python fitting_cli.py -s 2 -f 10 -i 115 -c alt_sched --sim_variability epoch --fit_variability epoch    -o validation/pwv_sim_epoch_fit_epoch.csv
