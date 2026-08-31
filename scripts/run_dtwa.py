"""Independent check of the cumulant closure for an inhomogeneous line.

The second-order cumulant closure is validated elsewhere against exact
solutions: against the exact master equation for eight distinguishable spins
(`tests/test_exact.py`) and against the permutation-invariant Dicke solver for
uniform ensembles up to eighty spins (`run_validation.py`).  Neither reaches an
inhomogeneously broadened ensemble at the spin numbers used in the article.

This script closes that gap with the discrete truncated Wigner approximation
(`cavsqueeze.dtwa`), which truncates the dynamics rather than the statistics
and is therefore independent of the closure.  Both solvers are given exactly
the same Hamiltonian: the line is discretized once, and the truncated Wigner
run places one classical spin at each class detuning for every member of that
class.  The comparison therefore isolates the closure, which is what is in
question; the discretization itself is checked separately in Tables S1 and S2.
The line is truncated at DELTA_MAX half widths, since spins further out are
free to the accuracy of the article's own spectator treatment and keeping them
would only make the classical trajectories stiff without testing anything.

Results -> data/dtwa.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from cavsqueeze import dtwa
from cavsqueeze.cumulant import Rates, evolve, wineland_xi2
from cavsqueeze.ensemble import Ensemble, lineshape, tail_resolved_classes
from cavsqueeze.protocols import css_x
from cavsqueeze.resonator import CavityParams

TWO_PI = 2.0 * np.pi
FW = TWO_PI * GAMMA_INH_HZ                 # line width in rad/s
SHAPES = ["voigt", "gaussian", "lorentzian"]
RATIOS = [2.0, 4.0, 8.0]                   # chi N / gamma_inh
N_TRAJ = 4000


DELTA_MAX = 20.0        # line truncated at this many half widths, both solvers


def shared_ensemble(chiN, shape, N):
    """One discretization of the line with an INTEGER number of spins per class.

    The truncated Wigner run needs whole spins, so the class populations are
    rounded to integers by the largest-remainder rule, which preserves the
    total.  The cumulant solver is then given the same integer populations, so
    that the two solvers see an identical Hamiltonian and the comparison is of
    the two truncations and nothing else.
    """
    dist = lineshape(shape, FW)
    ens = tail_resolved_classes(dist, N, M_core=48, M_tail=16, fwhm=FW,
                                core_edge=3.0 * FW, delta_max=DELTA_MAX * FW)
    n = np.asarray(ens.n, float)
    n = n * (N / n.sum())
    base = np.floor(n).astype(int)
    rem = int(round(N - base.sum()))
    if rem > 0:
        order = np.argsort(-(n - base))
        base[order[:rem]] += 1
    keep = base > 0
    return Ensemble(delta=np.asarray(ens.delta)[keep],
                    weight=np.asarray(ens.weight)[keep],
                    n=base[keep].astype(float))


def cumulant_curve(chiN, shape, N, t_eval):
    """Optimum-seeking curve of the Wineland parameter from the cumulant solver.

    No cavity loss: chi only, so that the comparison isolates twisting plus
    detunings, which is the part with no other reference.
    """
    ens = shared_ensemble(chiN, shape, N)
    # kappa = 0 makes chi = g^2/Delta and Gamma_SR = 0, so the comparison
    # isolates twisting plus detunings, the part with no other reference.
    p = CavityParams(g=np.sqrt(chiN / N), kappa=0.0, Delta=1.0)
    rt = Rates.from_params(p, ens)
    st = css_x(rt.M)
    states = evolve(st, rt, float(np.max(t_eval)), t_eval=t_eval, rtol=1e-9)
    out = []
    for s in states:
        xi2, _, _, _, _ = wineland_xi2(s, rt.n, None, spec_n=rt.spec_n)
        out.append(float(xi2))
    return np.array(out)


def dtwa_curve(chiN, shape, N, t_eval, seed, n_traj=4000):
    """The same discretized line, one classical spin per member of each class."""
    ens = shared_ensemble(chiN, shape, N)
    counts = np.round(ens.n).astype(int)
    delta = np.repeat(ens.delta, counts)
    G = np.repeat(ens.weight, counts)
    n_spins = len(delta)
    chi1 = chiN / float(np.sum(G ** 2))
    out = dtwa.evolve(delta, G, chi1, t_eval, n_traj=n_traj, seed=seed + 1)
    return np.array([dtwa.wineland(o["mean"], o["cov"], n_spins) for o in out]), n_spins


def main():
    rows, curves = [], {}
    # Scan N at a fixed interaction-to-linewidth ratio.  The closure is expected
    # to fail when Q^3/N is of order one, with Q the twisting strength at the
    # optimum; that is the regime these sizes sit in, and the article's
    # ensembles are at least a million times larger.
    plan = [(N, 4.0, "voigt") for N in (125, 250, 500, 1000)] + \
           [(1000, 4.0, s) for s in ("gaussian", "lorentzian")]
    for N, ratio, shape in plan:
        chiN = ratio * FW
        t_opt = 3.0 / chiN * N ** (1.0 / 3.0)
        t_eval = np.linspace(0.0, 1.3 * t_opt, 22)
        t0 = time.time()
        xc = cumulant_curve(chiN, shape, float(N), t_eval)
        xd, n_spins = dtwa_curve(chiN, shape, N, t_eval, seed=11, n_traj=N_TRAJ)
        kc, kd = int(np.nanargmin(xc)), int(np.nanargmin(xd))
        Q = chiN * t_eval[kc]
        dev = abs(dB(xc[kc]) - dB(xd[kd]))
        rows.append([n_spins, ratio, SHAPES.index(shape), Q, Q ** 3 / n_spins,
                     dB(xc[kc]), dB(xd[kd]), dev])
        key = f"{shape}_{N}_{ratio:g}"
        curves[key + "_t"] = t_eval
        curves[key + "_cum"] = xc
        curves[key + "_dtwa"] = xd
        print(f"  N={n_spins:5d}  {shape:11s} Q={Q:5.2f}  Q^3/N={Q**3/n_spins:6.3f}  "
              f"cumulant {dB(xc[kc]):7.2f} dB   trajectory {dB(xd[kd]):7.2f} dB   "
              f"|diff| {dev:.2f} dB   [{time.time()-t0:.0f} s]", flush=True)
    save("dtwa", rows=np.array(rows), shapes=np.array(SHAPES), delta_max=DELTA_MAX,
         n_traj=N_TRAJ, **curves)
    d = np.array(rows)
    print("\nlargest |difference|: %.2f dB at N=%d; smallest %.2f dB at N=%d"
          % (d[:, 7].max(), int(d[np.argmax(d[:, 7]), 0]),
             d[:, 7].min(), int(d[np.argmin(d[:, 7]), 0])))


if __name__ == "__main__":
    main()
