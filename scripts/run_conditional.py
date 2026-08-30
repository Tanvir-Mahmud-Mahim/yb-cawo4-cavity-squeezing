"""Twisting and measurement together: class-resolved conditional cumulant solver.

The measurement of J_z through the resonator is added to the second-order
cumulant solver as a Gaussian (Kalman) conditioning term with rate Gamma_m and
the corresponding back-action (cumulant.Rates.meas, meas_eta).  Every spin is a
class (no spectators), so every spin is conditioned.  At the superconducting
operating point the solver gives, for each line shape and probe photon number,
the conditional variance of J_z (to be compared with the Riccati model of
Sec. S7), and the best quadrature in the plane perpendicular to the mean spin,
which is what twisting and measurement together can deliver.
Outputs: data/conditional.npz.
"""
from common import *  # noqa
from cavsqueeze.cumulant import Rates, evolve, collective_moments, wineland_xi2
from cavsqueeze.protocols import css_x
from cavsqueeze.ensemble import tail_resolved_classes
from concurrent.futures import ProcessPoolExecutor

N = 1e10
KAPPA_HZ, GN_HZ, DELTA_HZ = 1e4, 1e6, 3e7
TS = np.geomspace(2e-5, 2e-3, 16)


def job(args):
    shape, frac, nb, eta = args
    p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, DELTA_HZ, T=0.02, T2=T2_SPIN)
    fw = TWO_PI * GAMMA_INH_HZ
    ens = tail_resolved_classes(lineshape("voigt", fw, frac), N, 24, 12, fwhm=fw, core_edge=3 * fw,
                                delta_max=30.0 * fw, spectator_beyond=None)
    rt = Rates.from_params(p, ens)
    rt.meas = 64 * eta * (p.g**2 / p.Delta) ** 2 * nb / p.kappa
    rt.meas_eta = eta
    sts = evolve(css_x(rt.M), rt, TS[-1], t_eval=TS, rtol=1e-7)
    rows = []
    for t, s in zip(TS, sts):
        J, Cov, S1, S2 = collective_moments(s, rt.n)
        xi2, ang, vmin, vmax, Jn = wineland_xi2(s, rt.n)
        C = 2 * np.hypot(J[0], J[1]) / N
        rows.append([t, C, Cov[2, 2], (N / 4) * C**2 / Cov[2, 2], 1 / xi2, ang])
    rows = np.array(rows)
    print(f"{shape} eta={eta} n_bar={nb:.0e}: gain_z best {dB(rows[:,3].max()):.2f} dB, best quadrature {dB(rows[:,4].max()):.2f} dB", flush=True)
    return (shape, nb, eta), rows


if __name__ == "__main__":
    jobs = [("voigt", 0.3, 0.0, 0.5), ("gaussian", 0.0, 0.0, 0.5), ("lorentzian", 1.0, 0.0, 0.5)]
    for shape, frac in [("voigt", 0.3), ("gaussian", 0.0), ("lorentzian", 1.0)]:
        for nb in [1e8, 1e9]:
            jobs.append((shape, frac, nb, 0.5))
    for eta in [0.8, 1.0]:
        for nb in [1e8, 1e9]:
            jobs.append(("voigt", 0.3, nb, eta))
    out = {"t": TS}
    with ProcessPoolExecutor(3) as ex, Timer("conditional"):
        for (shape, nb, eta), rows in ex.map(job, jobs):
            out[f"{shape}_n{int(np.log10(nb)) if nb else 0}_eta{eta}"] = rows
    save("conditional", **out)
