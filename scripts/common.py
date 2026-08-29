"""Shared settings for all figure scripts."""
from __future__ import annotations

import json
import os
import sys
import time

# single-threaded BLAS: the class matrices are small and worker processes must not
# oversubscribe the cores (thread contention slows the ODE integration by > 10x)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

from cavsqueeze import lineshape, from_hz, loop_gap_dispersive, homogeneous  # noqa: E402
from cavsqueeze.ensemble import tail_resolved_classes  # noqa: E402

TWO_PI = 2 * np.pi
GAMMA_INH_HZ = 5e3  # spin inhomogeneous linewidth of 171Yb:CaWO4 (FWHM), Tiranov et al. / Fukumori et al.
LORENTZ_FRACTION = 0.3  # Voigt lineshape used in the Supplement of Fukumori et al.
OMEGA_S_HZ = 3.08385e9
T2_SPIN = 0.15  # s, Tiranov et al.

# discretisation presets: (M_core, M_tail)
GRID_STD = (48, 16)
GRID_LIGHT = (24, 12)
GRID_SCAN = (32, 12)
SPECTATOR_FACTOR = 5.0  # spins with |delta| > factor x max(chi N, 2 FWHM) are treated as free spectators


def standard_ensemble(N, chiN, shape="voigt", grid=GRID_STD, fwhm_hz=GAMMA_INH_HZ, lorentz_fraction=LORENTZ_FRACTION):
    """Tail-resolved discretisation of the line; the far tail is capped at
    10 max(chi N, 2 FWHM) beyond which spins are free (see Supplement)."""
    fw = TWO_PI * fwhm_hz
    d = lineshape(shape, fw, lorentz_fraction)
    dmax = 1000.0 * fw
    free = SPECTATOR_FACTOR * max(abs(chiN), 2 * fw)
    return tail_resolved_classes(d, N, grid[0], grid[1], fwhm=fw, core_edge=3 * fw, delta_max=dmax, spectator_beyond=free)


def dB(x):
    return 10 * np.log10(np.asarray(x, float))


def save(name, **arrays):
    path = os.path.join(DATA, name + ".npz")
    np.savez(path, **arrays)
    print("saved", path)


def load(name):
    path = os.path.join(DATA, name + ".npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    return None


def save_json(name, obj):
    with open(os.path.join(DATA, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] {time.time() - self.t0:.1f} s", flush=True)
