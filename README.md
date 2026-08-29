# yb-cawo4-cavity-squeezing

Beyond-mean-field simulation of cavity-mediated spin squeezing (one-axis twisting)
for solid-state clock-transition ensembles, applied to 171Yb3+:CaWO4.

Companion code for: T. M. Mahim, M. M. Rahman, A. S. M. Mohsin, *Design limits of
cavity-mediated spin squeezing in a solid-state clock-transition ensemble: a
beyond-mean-field study of 171Yb3+:CaWO4*.

The package `cavsqueeze` implements

* the adiabatically eliminated Tavis-Cummings model with collective emission,
  collective thermal absorption, single-spin dephasing and coupling inhomogeneity
  (`resonator.py`);
* discretisation of Gaussian, Lorentzian and Voigt lines into frequency classes with
  tail resolution (`ensemble.py`);
* a class-resolved second-order cumulant expansion in *connected* variables that is
  exact to machine precision for arbitrarily large N (`cumulant.py`), together with
  the raw-moment form used as a test reference (`cumulant_raw.py`);
* exact references: QuTiP master equation for distinguishable spins and the
  permutation-invariant Dicke solver PIQS (`exact.py`);
* pulse sequences: echo twist, Ramsey, twist-untwist readout, plain squeezed readout
  (`protocols.py`);
* far-detuned spectator spins propagated analytically, which removes the stiffness of
  heavy-tailed lines (`ensemble.tail_resolved_classes`, `cumulant.evolve`).

## Installation

```
pip install numpy scipy matplotlib qutip pytest
pip install -e .          # or add the repository root to PYTHONPATH
pytest tests              # validation testbench (about 10 s)
```

## Reproducing the paper

```
cd scripts
python run_validation.py   # Fig. 2 data (a few minutes)
python run_benchmark.py    # Fig. 3 data
python run_loopgap.py      # Fig. 4 data
python run_scaling.py      # Fig. 5 data
python run_designmap.py    # Fig. 6 data
python run_inhomog.py      # Fig. 7(a) data
python run_readout.py      # Fig. 7(b,c) data
python make_figures.py        # data figures (Figs. 2 to 7) -> figures/
python make_device_figure.py  # 3-D device schematic (Fig. 1) -> figures/fig_device.*

# Figures use the Times New Roman font. If it is not installed, Matplotlib falls back
# to DejaVu Sans; install the font (for example from a Windows machine, C:\Windows\Fonts)
# into ~/.fonts and clear ~/.cache/matplotlib to reproduce the published look.
python extract_numbers.py  # every number quoted in the paper -> data/numbers.json
```
`run_all.sh` chains the data scripts (about 6 hours on two cores; set
`OPENBLAS_NUM_THREADS=1`, which the scripts do automatically, or the workers oversubscribe
the cores). All datasets, the extracted numbers and the figures are archived on Zenodo:
https://doi.org/10.5281/zenodo.22148969 (all versions; v1.1.0: https://doi.org/10.5281/zenodo.22157408)

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
