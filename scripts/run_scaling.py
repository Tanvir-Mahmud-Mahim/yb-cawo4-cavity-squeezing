"""Universal scaling of the optimum squeezing.
 (a) xi^2_opt versus chi N / gamma_inh at fixed 2 Delta/kappa (100, 1000, 10000), Voigt line;
     Gaussian and Lorentzian at 2 Delta/kappa = 1000.
 (b) xi^2_opt versus cavity bath temperature (thermal photons) at 2 Delta/kappa = 66.7 and 6000.
 (c) xi^2_opt versus single-spin T2.
"""
from common import *  # noqa
from cavsqueeze.protocols import optimal_squeezing
from concurrent.futures import ProcessPoolExecutor

N = 1e10
KAPPA = 1e4


def job_a(args):
    ratio, chiN_hz, shape, echo = args
    Delta = ratio * KAPPA / 2
    gN = np.sqrt(chiN_hz * Delta)  # chi N = (g sqrt N)^2 / Delta in the dispersive limit
    p = from_hz(gN / np.sqrt(N), KAPPA, Delta, T=0.0, T2=T2_SPIN)
    ens = standard_ensemble(N, p.chi * N, shape, GRID_SCAN)
    best = optimal_squeezing(p, ens, 2e-6, 2e-3, echo=echo, rtol=1e-6, n_coarse=10, max_fine=24)
    return ratio, chiN_hz, shape, best["xi2"], best["t"], best["Q"], best["contrast"], gN, echo


def job_b(args):
    ratio, T = args
    echo = True
    chiN_hz = 20 * GAMMA_INH_HZ
    Delta = ratio * KAPPA / 2
    gN = np.sqrt(chiN_hz * Delta)
    p = from_hz(gN / np.sqrt(N), KAPPA, Delta, T=T, T2=T2_SPIN)
    ens = standard_ensemble(N, p.chi * N, "voigt", GRID_SCAN)
    best = optimal_squeezing(p, ens, 2e-6, 2e-3, echo=echo, rtol=1e-6, n_coarse=10, max_fine=24)
    return ratio, T, p.n_th, best["xi2"], best["t"]


def job_c(args):
    ratio, T2 = args
    echo = True
    chiN_hz = 20 * GAMMA_INH_HZ
    Delta = ratio * KAPPA / 2
    gN = np.sqrt(chiN_hz * Delta)
    p = from_hz(gN / np.sqrt(N), KAPPA, Delta, T=0.0, T2=T2)
    ens = standard_ensemble(N, p.chi * N, "voigt", GRID_SCAN)
    best = optimal_squeezing(p, ens, 2e-6, 2e-3, echo=echo, rtol=1e-6, n_coarse=10, max_fine=24)
    return ratio, T2, best["xi2"], best["t"]


if __name__ == "__main__":
    out = {}
    chiN_ratios = np.array([0.5, 1, 2, 4, 8, 16])
    jobs = [(r, c * GAMMA_INH_HZ, "voigt", True) for r in [100, 1000, 10000] for c in chiN_ratios]
    jobs += [(1000, c * GAMMA_INH_HZ, "voigt", False) for c in chiN_ratios]
    jobs += [(1000, c * GAMMA_INH_HZ, s, True) for s in ["gaussian", "lorentzian"] for c in chiN_ratios]
    rows = []
    with ProcessPoolExecutor(2) as ex, Timer("scaling a"):
        for r in ex.map(job_a, jobs):
            print(r, flush=True)
            rows.append([r[0], r[1], {"voigt": 0, "gaussian": 1, "lorentzian": 2}[r[2]], r[3], r[4], r[5], r[6], r[7], 1 if r[8] else 0])
    out["a_rows"] = np.array(rows)
    save("scaling", **out)
    Ts = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5])
    jobs = [(r, T) for r in [66.7, 6000] for T in Ts]
    rows = []
    with ProcessPoolExecutor(2) as ex, Timer("scaling b"):
        for r in ex.map(job_b, jobs):
            print(r, flush=True)
            rows.append(list(r))
    out["b_rows"] = np.array(rows)
    save("scaling", **out)
    T2s = np.array([1e-3, 3e-3, 10e-3, 30e-3, 0.15])
    jobs = [(r, T2) for r in [66.7, 6000] for T2 in T2s]
    rows = []
    with ProcessPoolExecutor(2) as ex, Timer("scaling c"):
        for r in ex.map(job_c, jobs):
            print(r, flush=True)
            rows.append(list(r))
    out["c_rows"] = np.array(rows)
    save("scaling", **out)
