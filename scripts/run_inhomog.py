"""Coupling inhomogeneity: spins with log-uniformly distributed couplings over a
dynamic range D (mean weight one).  The cavity squeezes the coupling-weighted
collective spin; an optical population readout measures the unweighted one.
Operating point: N = 1e10, kappa/2pi = 10 kHz, mean g sqrt N = 1 MHz, Delta/2pi = 30 MHz, 20 mK.
"""
from common import *  # noqa
from cavsqueeze.protocols import squeezing_after
from cavsqueeze.ensemble import log_uniform_weights, Ensemble
from concurrent.futures import ProcessPoolExecutor

N = 1e10
p = from_hz(1e6 / np.sqrt(N), 1e4, 30e6, T=0.02, T2=T2_SPIN)
Ds = np.array([1, 2, 4, 10, 30, 100], float)
t_list = np.geomspace(1e-5, 1e-3, 12)
K = 5


def build(D):
    base = standard_ensemble(1.0, p.chi * N, "voigt", (16, 8))
    w, pw = log_uniform_weights(D, K)
    d = np.repeat(base.delta, len(w))
    ww = np.tile(w, base.M)
    n = np.repeat(base.n, len(w)) * np.tile(pw, base.M) * N
    # spectators (0.3% of the spins here) carry unit weight; their contribution to the
    # weighted normalisation is negligible at this level
    return Ensemble(delta=d, weight=ww, n=n, spec_delta=base.spec_delta, spec_n=base.spec_n * N)


def job(D):
    ens = build(D)
    xi_w, xi_u, con = [], [], []
    for t in t_list:
        r_w = squeezing_after(p, ens, t, echo=True, weights=ens.weight, rtol=1e-6)
        r_u = squeezing_after(p, ens, t, echo=True, weights=None, rtol=1e-6)
        xi_w.append(r_w["xi2"])
        xi_u.append(r_u["xi2"])
        con.append(r_u["contrast"])
    S2 = float(np.sum(ens.n * ens.weight**2) + np.sum(ens.spec_n)) / N
    return D, np.array(xi_w), np.array(xi_u), np.array(con), S2


if __name__ == "__main__":
    out = dict(D=Ds, t=t_list, N=N)
    with ProcessPoolExecutor(2) as ex, Timer("inhomogeneity"):
        for D, xw, xu, con, S2 in ex.map(job, Ds):
            print(D, "weighted best", dB(xw.min()), "unweighted best", dB(xu.min()), "sum w^2/N", S2, flush=True)
            out[f"xi_w_{int(D)}"] = xw
            out[f"xi_u_{int(D)}"] = xu
            out[f"con_{int(D)}"] = con
            out[f"S2_{int(D)}"] = S2
            save("inhomog", **out)
