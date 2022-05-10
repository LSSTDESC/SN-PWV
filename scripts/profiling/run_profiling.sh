# Pipeline run for an entire cadence

TOTAL_PROC=63
MIN_ALLOC=2
MAX_WRITE=5
NUM_LC_SIMS=10000

for ((NUM_WRITE = 1; NUM_WRITE <= MAX_WRITE; NUM_WRITE++)); do
  REMAINING_PROC=$((TOTAL_PROC - NUM_WRITE - MIN_ALLOC))

  for ((NUM_FIT = MIN_ALLOC; NUM_FIT <= REMAINING_PROC; NUM_FIT += 2)); do
    NUM_SIM=$((TOTAL_PROC - NUM_WRITE - NUM_FIT))
    
    echo $NUM_SIM $NUM_FIT $NUM_WRITE

    rm -rf ./temp.*.h5
    { time python ../../snat_sim_cli.py \
        --overwrite -i $NUM_LC_SIMS \
        -s $NUM_SIM -f $NUM_FIT -w $NUM_WRITE \
        -c alt_sched_rolling --sim_variability epoch --fit_variability epoch \
        -o ./temp.h5 ; } 2>  "$NUM_SIM.$NUM_FIT.$NUM_WRITE.txt"
  done
done