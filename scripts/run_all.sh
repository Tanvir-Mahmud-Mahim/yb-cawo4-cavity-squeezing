#!/bin/bash
# Regenerates every dataset in data/ (then run make_figures.py).
cd "$(dirname "$0")"
for s in run_benchmark run_loopgap run_scaling run_designmap run_inhomog run_readout; do
  echo "=== $s ===" ; python $s.py > ../data/log_$s.txt 2>&1 ; tail -3 ../data/log_$s.txt
done
