"""Monte Carlo uncertainty of the trajectory check, measured rather than assumed.

`run_dtwa.py` reports one trajectory number per row, from one seed.  That number
carries a sampling error, and the difference between the two solvers is only
meaningful against it.  This script repeats two rows of that table with
independent seeds and reports the spread, and it measures the deviation from the
exact one-axis-twisting optimum instead of quoting the threshold the unit test
enforces.

Rows repeated: the smallest ensemble, where the seed-to-seed spread is largest,
and the row with the smallest difference between the solvers, where the spread
matters most for the conclusion.

Results -> data/dtwa_seeds.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

from cavsqueeze import dtwa
from run_dtwa import FW, N_TRAJ, RATIO, dtwa_curve

SEEDS = [11, 12, 13, 14]


def kitagawa_ueda(N, mu):
    """Exact Wineland parameter for one-axis twisting, Kitagawa and Ueda (1993)."""
    A = 1.0 - np.cos(mu) ** (N - 2)
    B = 4.0 * np.sin(mu / 2.0) * np.cos(mu / 2.0) ** (N - 2)
    var = 1.0 + 0.25 * (N - 1) * (A - np.hypot(A, B))
    return var / np.cos(mu / 2.0) ** (2 * (N - 1))


def ku_deviation(N=200, n_traj=4000, seed=1):
    """Deviation of the trajectory optimum from the exact one, in dB."""
    t = np.linspace(0.0, 0.06, 25)
    out = dtwa.evolve(np.zeros(N), np.ones(N), 1.0, t, n_traj=n_traj, seed=seed)
    got = min(dtwa.wineland(o["mean"], o["cov"], N) for o in out)
    want = min(kitagawa_ueda(N, 2.0 * tt) for tt in t)
    return abs(dB(got) - dB(want))


def main():
    dev = ku_deviation()
    print("one-axis-twisting deviation from exact: %.3f dB" % dev, flush=True)

    rows, out = [], {}
    for N, shape in ((125, "voigt"), (1000, "lorentzian")):
        vals = []
        for sd in SEEDS:
            t0 = time.time()
            chiN = RATIO * FW
            t_opt = 3.0 / chiN * N ** (1.0 / 3.0)
            t_eval = np.linspace(0.0, 1.3 * t_opt, 22)
            xd, n_spins = dtwa_curve(chiN, shape, N, t_eval, seed=sd, n_traj=N_TRAJ)
            v = dB(np.nanmin(xd))
            vals.append(v)
            print("  N=%5d %-11s seed %2d -> %7.3f dB   [%.0f s]"
                  % (n_spins, shape, sd, v, time.time() - t0), flush=True)
        vals = np.array(vals)
        sd_ = float(vals.std(ddof=1))
        print("  N=%5d %-11s mean %7.3f  sd %.3f  spread %.3f dB\n"
              % (N, shape, vals.mean(), sd_, vals.max() - vals.min()), flush=True)
        rows.append([N, vals.mean(), sd_, vals.max() - vals.min()])
        out["vals_%s_%d" % (shape, N)] = vals

    save("dtwa_seeds", rows=np.array(rows), seeds=np.array(SEEDS),
         n_traj=N_TRAJ, ku_dev=dev, **out)


if __name__ == "__main__":
    main()
