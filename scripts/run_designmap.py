"""Superconducting-resonator design map for 171Yb:CaWO4 (N = 1e10 spins, Voigt line,
gamma_inh/2pi = 5 kHz, T = 20 mK, T2 = 150 ms).
For each (kappa, g sqrt N) the cavity detuning is chosen from the candidate set
Delta = (g sqrt N)^2 / (nu gamma_inh), nu in {4, 8, 16}, subject to the dispersive
condition Delta >= 5 g sqrt N, and the interaction time is optimized.  The best
(Delta, t) pair and the resulting xi^2 are stored.
"""
from common import *  # noqa
from cavsqueeze.protocols import optimal_squeezing
from concurrent.futures import ProcessPoolExecutor

N = 1e10
T_BATH = 0.02
kappas = np.array([3e3, 1e4, 3e4, 1e5, 3e5])
gNs = np.array([0.1e6, 0.2e6, 0.5e6, 1e6, 2e6, 5e6])
nus = np.array([4, 8, 16], float)


def job(args):
    kappa, gN = args
    best = None
    rows = []
    for nu in nus:
        Delta = max(gN**2 / (nu * GAMMA_INH_HZ), 5 * gN)
        p = from_hz(gN / np.sqrt(N), kappa, Delta, T=T_BATH, T2=T2_SPIN)
        ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
        b = optimal_squeezing(p, ens, 2e-6, 2e-3, echo=True, rtol=1e-6, n_coarse=10, max_fine=24)
        rows.append([kappa, gN, nu, Delta, b["xi2"], b["t"], b["Q"], p.chi * N / TWO_PI, p.Gamma_SR * N / TWO_PI])
        if best is None or b["xi2"] < best[4]:
            best = rows[-1]
    return best, rows


if __name__ == "__main__":
    jobs = [(k, g) for k in kappas for g in gNs]
    best_rows, all_rows = [], []
    with ProcessPoolExecutor(2) as ex, Timer("design map"):
        for best, rows in ex.map(job, jobs):
            print("best", best, flush=True)
            best_rows.append(best)
            all_rows += rows
            save("designmap", kappas=kappas, gNs=gNs, nus=nus, N=N, best=np.array(best_rows), all=np.array(all_rows))
