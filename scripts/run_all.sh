#!/bin/bash
# Regenerates every dataset in data/ (then run make_figures.py).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in run_benchmark run_loopgap run_scaling run_designmap run_inhomog run_readout run_elimination run_reversal run_robustness run_measurement; do
  echo "=== $s ===" ; python $s.py > ../data/log_$s.txt 2>&1 ; tail -3 ../data/log_$s.txt
done
