# Runs a subset of light-curves through the data analysis pipeline using different global configuration parameters
# Intended for use during pipeline verification/validation

# Validation runs to ensure atmospheric variability is propagating in the expected ways
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0     --fit_variability 0        -o ../../data/validation/pwv_sim_0_fit_0.h5
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 4     --fit_variability 4        -o ../../data/validation/pwv_sim_4_fit_4.h5
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0     --fit_variability 4        -o ../../data/validation/pwv_sim_0_fit_4.h5
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability 4        -o ../../data/validation/pwv_sim_epoch_fit_4.h5
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability seasonal -o ../../data/validation/pwv_sim_epoch_fit_seasonal.h5
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability epoch --fit_variability epoch    -o ../../data/validation/pwv_sim_epoch_fit_epoch.h5

# Runs with light-curve scatter turned off and a fixed SNR
# Useful in checking handling of the underlying model covariances
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0 --fit_variability 0 -o ../../data/validation/no_scat_pwv_sim_0_fit_0.h5         --no-scatter
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0 --fit_variability 0 -o ../../data/validation/snr1000_pwv_sim_0_fit_0.h5                      --fixed-snr 1000
python ../../snat_sim_cli.py --overwrite -s 1 -f 10 -i 500 -c alt_sched --sim_variability 0 --fit_variability 0 -o ../../data/validation/no_scat_snr1000_pwv_sim_0_fit_0.h5 --no-scatter --fixed-snr 1000
