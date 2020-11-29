export CADENCE_SIMS='/home/djperrefort/Github/SN-PWV/scripts/profiling/data'
python fitting_cli.py -c alt_sched -s 0 -f 0 -o validation/pwv_sim_0_fit_0.csv -i 100 -p 12
python fitting_cli.py -c alt_sched -s 4 -f 4 -o validation/pwv_sim_4_fit_4.csv -i 100 -p 12
python fitting_cli.py -c alt_sched -s epoch -f 4 -o validation/pwv_sim_epoch_fit_4.csv -i 100 -p 12
