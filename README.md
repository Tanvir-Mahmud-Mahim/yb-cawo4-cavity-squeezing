# yb-cawo4-cavity-squeezing

Beyond-mean-field simulation of cavity-mediated spin squeezing (one-axis twisting)
for solid-state clock-transition ensembles, applied to 171Yb3+:CaWO4, and of
squeezing by measurement of the spin population through the resonator.

Companion code for: T. M. Mahim, M. M. Rahman, A. S. M. Mohsin, *Synchronization
sets the coherence and the squeezing limit of a spin ensemble in a cavity*.

The package `cavsqueeze` implements

* the adiabatically eliminated Tavis-Cummings model with collective emission,
  collective thermal absorption, single-spin dephasing and coupling inhomogeneity
  (`resonator.py`);
* discretization of Gaussian, Lorentzian and Voigt lines into frequency classes with
  tail resolution (`ensemble.py`);
* a class-resolved second-order cumulant expansion in *connected* variables, whose
  cost is set by the number of classes rather than by N, so that N = 1e15 is no
  harder than N = 1e3, and which loses no precision to cancellation at large N
  because the means are subtracted analytically (`cumulant.py`), together with
  the raw-moment form used as a test reference (`cumulant_raw.py`); the solver can
  condition the ensemble on a continuous measurement of J_z through the resonator
  (`Rates.meas`, `Rates.meas_eta`);
* exact references: QuTiP master equation for distinguishable spins and the
  permutation-invariant Dicke solver PIQS (`exact.py`);
* pulse sequences: echo twist, Ramsey, twist-untwist readout, plain squeezed readout
  (`protocols.py`);
* far-detuned spectator spins propagated analytically, which removes the stiffness of
  heavy-tailed lines (`ensemble.tail_resolved_classes`, `cumulant.evolve`).
* an independent solver for cross-checking: discrete truncated Wigner trajectories
  (`dtwa.py`), which truncate the equations of motion rather than the statistics.

The mean-field limit of the same equations reduces to a pendulum for each spin in the
field of the collective spin, whose conserved energy closes into a self-consistency
equation for the Ramsey contrast; `scripts/run_locking.py` derives, tests and solves it.

## Installation

```
pip install numpy scipy matplotlib qutip pytest
pip install -e .          # or add the repository root to PYTHONPATH
pytest tests              # validation testbench (about 2 minutes)
```

## Reproducing the paper

```
cd scripts
python run_validation.py   # Fig. S1 (fig:validation) data (a few minutes)
python run_benchmark.py    # Fig. 2 (fig:benchmark) data
python run_scaling.py      # Fig. S2 (fig:scaling) data, and the interaction scans of Fig. 3 (fig:designmap)(b,c)
python run_designmap.py    # Fig. 3 (fig:designmap)(d) and Fig. S3 (fig:designmap_extra) data
python run_loopgap.py      # Fig. S4 (fig:loopgap) data
python run_inhomog.py      # Fig. S5 (fig:inhomog_readout)(a) data
python run_readout.py      # Fig. S5 (fig:inhomog_readout)(b,c) data
python run_elimination.py  # Fig. S6 (fig:beyond)(a) data: resonator kept as a quantum mode, no rotating-wave approximation
python run_reversal.py     # Fig. S6 (fig:beyond)(b) data: ring-down at the detuning reversal of the twist-untwist readout
python run_robustness.py   # Fig. S6 (fig:beyond)(c,d) data: line shape, T2, finite and imperfect pulses
python run_measurement.py  # Fig. S7 (fig:measure) data: squeezing by measurement through the resonator (about 1 hour)
python run_conditional.py  # supplement Table S11 (tab:cond): twisting and measurement together, conditional cumulant solver (about 30 min)
python run_echo.py         # Fig. 4 (fig:echo) data: spin echo against the interaction (about 20 min); then run_echo_extra.py (tau scans, no-emission check)
python run_locking.py      # Fig. 3 (fig:designmap)(a) data: the closed law for the synchronization order parameter, its threshold, and the orbit average behind it (about 20 min)
python run_decompose.py    # splits the noise at the optimum into locked core and unlocked wings; the two-limit law of Eq. (9) (eq:twolimits) is tested on this scan (about 1 hour)
python run_dtwa.py         # independent check of the cumulant closure for an inhomogeneous line, against discrete truncated Wigner trajectories (about 30 min)
python run_dtwa_seeds.py   # sampling error of that check, from repeated seeds, and its deviation from the exact one-axis-twisting optimum (about 1 hour)
python hyperfine_levels.py # zero-field hyperfine levels and Sz/Sx matrix elements -> data/hyperfine_levels.json
python make_figures.py        # data figures (main Figs. 2 to 4 and Figs. S1 to S7) -> figures/
python make_device_figure.py  # 3-D device schematic, Fig. 1 (fig:device) -> figures/fig_device.*

# Figures use the Times New Roman font. If it is not installed, Matplotlib falls back
# to DejaVu Sans; install the font (for example from a Windows machine, C:\Windows\Fonts)
# into ~/.fonts and clear ~/.cache/matplotlib to reproduce the published look.
python extract_numbers.py  # every number quoted in the paper and supplement -> data/numbers.json
python check_consistency.py # title, version, DOI, citations and cross-references agree across files
```
`run_all.sh` chains the data scripts (about 7 hours on two cores; set
`OPENBLAS_NUM_THREADS=1`, which the scripts do automatically, or the workers oversubscribe
the cores). All datasets, the extracted numbers and the figures are archived on Zenodo:
https://doi.org/10.5281/zenodo.22148969 (concept DOI, always the latest version; v1.6.1 is https://doi.org/10.5281/zenodo.22212983)

## Minimal example

```python
import numpy as np
from cavsqueeze import from_hz, homogeneous
from cavsqueeze.protocols import optimal_squeezing
N = 1e10
p = from_hz(g_hz=1e6/np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
best = optimal_squeezing(p, homogeneous(N), 1e-6, 1e-2)
print(10*np.log10(best["xi2"]), "dB at", best["t"], "s")
```

## License

Apache-2.0 (see LICENSE). Please cite the paper if you use this code.
