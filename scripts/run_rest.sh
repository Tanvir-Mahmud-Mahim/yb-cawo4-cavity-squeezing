#!/bin/bash
cd "$(dirname "$0")"
for s in run_loopgap run_scaling run_designmap run_inhomog run_readout run_convergence; do
  echo "=== $s ===" ; python $s.py > ../data/log_$s.txt 2>&1 ; tail -3 ../data/log_$s.txt
done
