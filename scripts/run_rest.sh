#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in run_inhomog run_readout run_convergence; do
  echo "=== $s ===" ; python $s.py > ../data/log_$s.txt 2>&1 ; tail -3 ../data/log_$s.txt
done
