The `run_profiling.sh` scripts executes the analysis pipeline several times. 
Each run is executed with a different number of processes allocated to each 
stage of the analysis process: Simulating light-curves, fitting simulations, 
and writing results to disk (Note that there is also 1 process that is always 
allocated to reading in data from disk at the beginning of the pipeline).

The execution time of each pipline run are written to a text file named
using the number of processes allocated to each stage 
`$NUM_SIM.$NUM_FIT.$NUM_WRITE.txt`.

The `plot_results.py` script plots the runtime results and prints the best
runtime(s).

To fully understand the results, make sure to check the entire set of 
arguments passed to the analysis pipeline in `run_profiling.sh`.
