#!/bin/bash
# Regenerates every dataset in data/, then the extracted numbers and the
# cross-file consistency check.  Figures are made separately, with
# make_figures.py and make_device_figure.py, since they only read data/.
# About 7 hours on two cores.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Validation first: if the solver does not reproduce the exact solutions there
# is no point running anything else.
for s in run_validation run_convergence run_pulse_check \
         run_benchmark run_loopgap run_scaling run_designmap \
         run_inhomog run_readout run_elimination run_reversal run_robustness \
         run_measurement run_conditional run_echo run_echo_extra \
         run_locking run_decompose run_dtwa run_dtwa_seeds; do
  echo "=== $s ===" ; python $s.py > ../data/log_$s.txt 2>&1 ; tail -3 ../data/log_$s.txt
done

echo "=== hyperfine_levels ==="; python hyperfine_levels.py
echo "=== extract_numbers ===";  python extract_numbers.py
echo "=== check_consistency ==="; python check_consistency.py
