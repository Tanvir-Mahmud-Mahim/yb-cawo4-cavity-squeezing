"""The demonstrated loop-gap device (Fukumori et al.): where does the squeezing go?
 (a) xi^2 versus interaction time at N0 = 6e14 for homogeneous / Gaussian / Voigt /
     Lorentzian lines, echo and no-echo.
 (b) optimum xi^2 versus cavity detuning Delta at fixed hardware for N0 = 6e14 and 1.35e15.
 (c) optimum xi^2 versus N0 at Delta/2pi = 22 MHz.
Cavity bath temperature 80 mK (n_th = 0.19), T2 = 150 ms.
"""
from common import *  # noqa
from cavsqueeze.protocols import squeezing_after, optimal_squeezing
from concurrent.futures import ProcessPoolExecutor

N0 = 6e14
T_BATH = 0.08
p0, _ = loop_gap_dispersive(N0, T=T_BATH, T2=T2_SPIN)
t_list = np.unique(np.concatenate([np.geomspace(1e-5, 1.5e-3, 12), np.linspace(2e-5, 1.5e-3, 50)]))


def trace_job(args):
    shape, echo = args
    if shape == "homogeneous":
        ens = homogeneous(N0)
    else:
        ens = standard_ensemble(N0, p0.chi * N0, shape, GRID_STD)
    xi, con = [], []
    for t in t_list:
        r = squeezing_after(p0, ens, t, echo=echo, rtol=1e-6)
        xi.append(r["xi2"])
        con.append(r["contrast"])
    return shape, echo, np.array(xi), np.array(con)


def opt_job(args):
    N, Delta_hz, shape, echo = args
    p = from_hz(15e-3, 660e3, Delta_hz, T=T_BATH, T2=T2_SPIN)
    ens = homogeneous(N) if shape == "homogeneous" else standard_ensemble(N, p.chi * N, shape, GRID_SCAN)
    best = optimal_squeezing(p, ens, 1e-5, 2e-3, echo=echo, rtol=1e-6, n_coarse=10, max_fine=24)
    return N, Delta_hz, shape, best["xi2"], best["t"], best["contrast"], p.chi * N / TWO_PI, echo


if __name__ == "__main__":
    out = dict(t_list=t_list, N0=N0)
    jobs = [(s, e) for s in ["homogeneous", "gaussian", "voigt", "lorentzian"] for e in [True, False]]
    with ProcessPoolExecutor(2) as ex, Timer("traces"):
        for shape, echo, xi, con in ex.map(trace_job, jobs):
            out[f"a_xi_{shape}_{'echo' if echo else 'noecho'}"] = xi
            out[f"a_con_{shape}_{'echo' if echo else 'noecho'}"] = con
            print(shape, echo, "best", dB(xi.min()), "dB at", t_list[xi.argmin()])
    save("loopgap", **out)
    Deltas = np.array([3e6, 6e6, 12e6, 22e6, 44e6, 88e6])
    jobs = [(N, D, s, e) for N in [6e14, 1.35e15] for D in Deltas for s in ["homogeneous", "voigt"] for e in [True, False]]
    res = []
    with ProcessPoolExecutor(2) as ex, Timer("detuning scan"):
        for r in ex.map(opt_job, jobs):
            print(r, flush=True)
            res.append(r)
    out["b_Delta"] = Deltas
    out["b_rows"] = np.array([[r[0], r[1], 0 if r[2] == "homogeneous" else 1, r[3], r[4], r[5], r[6], 1 if r[7] else 0] for r in res])
    save("loopgap", **out)
    Ns = np.array([1e14, 2e14, 4e14, 6e14, 1.35e15, 3e15])
    jobs = [(N, 22e6, s, e) for N in Ns for s in ["homogeneous", "voigt"] for e in [True, False]]
    res = []
    with ProcessPoolExecutor(2) as ex, Timer("N scan"):
        for r in ex.map(opt_job, jobs):
            print(r, flush=True)
            res.append(r)
    out["c_N"] = Ns
    out["c_rows"] = np.array([[r[0], r[1], 0 if r[2] == "homogeneous" else 1, r[3], r[4], r[5], r[6], 1 if r[7] else 0] for r in res])
    save("loopgap", **out)
