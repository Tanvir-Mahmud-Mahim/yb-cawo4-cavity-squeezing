"""Sensitivity of the predictions to what the model does not fix.

 (a) Line shape.  Optimum squeezing of the loop-gap device (N0 = 6e14,
     Delta/2pi = 22 MHz) versus the Lorentzian fraction of a Voigt line of
     fixed total width 5 kHz, for the echo and the free-twisting sequences.  For
     each line the 1/e time of the free-induction decay is also computed, so
     that a measured Ramsey decay can be mapped onto a Lorentzian fraction.
 (b) Spin dephasing.  Optimum squeezing of the loop-gap device (Voigt line)
     versus the single-spin coherence time T2 from 0.1 ms to 150 ms.
 (c) Pulse imperfections.  Optimum squeezing with a pi pulse of finite
     duration (the cavity-mediated interaction stays on during the pulse),
     with a global rotation-angle error, and with a class-dependent rotation
     angle caused by the spread of the drive field (Gauss-Hermite three-point
     model of a 2% standard deviation, which also spreads the coupling), for the
     loop-gap device and for the superconducting operating point
     (N = 1e10, kappa/2pi = 10 kHz, g sqrt N/2pi = 1 MHz, Delta/2pi = 30 MHz, 20 mK).
     The free-twisting sequence, which has no pi pulse, is computed alongside.
"""
from common import *  # noqa
from cavsqueeze.cumulant import Rates, wineland_xi2, coherence, evolve, rotate, rotate_classes
from cavsqueeze.protocols import css_x, twist, twist_imperfect, pulse, optimal_squeezing, X, Y
from cavsqueeze.ensemble import Ensemble, voigt
from cavsqueeze.cumulant import product_state
from concurrent.futures import ProcessPoolExecutor

N_LG = 6e14
P_LG, _ = loop_gap_dispersive(N_LG, T=0.08, T2=T2_SPIN)
N_SC = 1e10
P_SC = from_hz(1e6 / np.sqrt(N_SC), 1e4, 30e6, T=0.02, T2=T2_SPIN)


def fid_1e_time(fwhm_hz, frac):
    """1/e time of the free-induction decay of a Voigt line (Lorentzian fraction frac)."""
    v = voigt(TWO_PI * fwhm_hz, frac)
    sig, gam = v.sigma, v.gamma  # rad/s
    # exp(-gam t - sig^2 t^2 / 2) = 1/e
    return 2.0 / (gam + np.sqrt(gam**2 + 2 * sig**2))  # stable form of the positive root


def best_of(p, ens, echo, t_lo=1e-5, t_hi=2e-3, **kw):
    b = optimal_squeezing(p, ens, t_lo, t_hi, echo=echo, rtol=1e-6, n_coarse=10, max_fine=24, **kw)
    return b["xi2"], b["t"]


def opt_generic(p, ens, seq_fn, t_lo, t_hi, n_coarse=10, max_fine=24, weights=None):
    """Same two-stage time optimisation as optimal_squeezing for an arbitrary sequence."""
    rt = Rates.from_params(p, ens)

    def f(t):
        st = seq_fn(rt, t)
        return wineland_xi2(st, rt.n, weights, spec_n=rt.spec_n)[0]

    ts = np.geomspace(t_lo, t_hi, n_coarse)
    vals = np.array([f(t) for t in ts])
    k = int(np.argmin(vals))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    period = 2 * np.pi / max(abs(rt.chiN), 1e-30)
    n_fine = int(np.clip(np.ceil((hi - lo) / period * 8), 8, max_fine))
    tf = np.linspace(lo, hi, n_fine)
    vf = np.array([f(t) for t in tf])
    j = int(np.nanargmin(np.concatenate([vals, vf])))
    all_t = np.concatenate([ts, tf])
    return float(np.concatenate([vals, vf])[j]), float(all_t[j])


def job_a(args):
    frac, echo = args
    ens = standard_ensemble(N_LG, P_LG.chi * N_LG, "voigt", GRID_SCAN, lorentz_fraction=frac)
    xi, t = best_of(P_LG, ens, echo)
    return "a", frac, echo, xi, t


def job_b(args):
    T2, echo = args
    p = loop_gap_dispersive(N_LG, T=0.08, T2=T2)[0]
    ens = standard_ensemble(N_LG, p.chi * N_LG, "voigt", GRID_SCAN)
    xi, t = best_of(p, ens, echo)
    return "b", T2, echo, xi, t


def job_c(args):
    label, kind, value = args
    p, N, grid = (P_LG, N_LG, GRID_SCAN) if label == "lg" else (P_SC, N_SC, GRID_LIGHT)
    t_lo, t_hi = (1e-5, 2e-3) if label == "lg" else (5e-6, 5e-4)
    ens = standard_ensemble(N, p.chi * N, "voigt", grid)
    if kind == "duration":
        # finite pi/2 preparation pulse (from the ground state) and finite pi pulse
        def seq(rt, t):
            st = product_state(rt.M, [0.0, 0.0, -1.0], K_spec=rt.K)
            st = pulse(st, rt, Y, np.pi / 2, duration=value, rtol=1e-6)
            return twist_imperfect(st, rt, t, echo=True, pi_duration=value, rtol=1e-6)
        xi, t = opt_generic(p, ens, seq, t_lo, t_hi)
    elif kind == "duration_noecho":
        def seq(rt, t):
            st = product_state(rt.M, [0.0, 0.0, -1.0], K_spec=rt.K)
            st = pulse(st, rt, Y, np.pi / 2, duration=value, rtol=1e-6)
            return evolve(st, rt, t, rtol=1e-6)
        xi, t = opt_generic(p, ens, seq, t_lo, t_hi)
    elif kind == "angle":
        def seq(rt, t):
            st = evolve(css_x(rt.M), rt, t / 2, rtol=1e-6)
            st = rotate(st, X, np.pi * (1 + value))
            return evolve(st, rt, t / 2, rtol=1e-6)
        xi, t = opt_generic(p, ens, seq, t_lo, t_hi)
    elif kind == "spread":
        # three-point Gauss-Hermite model of a field spread with standard deviation `value`
        w = np.array([1 - np.sqrt(3) * value, 1.0, 1 + np.sqrt(3) * value])
        pw = np.array([1, 4, 1]) / 6.0
        d = np.repeat(ens.delta, 3)
        ww = np.tile(w, ens.M)
        n = np.repeat(ens.n, 3) * np.tile(pw, ens.M)
        ens3 = Ensemble(delta=d, weight=ww, n=n, spec_delta=ens.spec_delta, spec_n=ens.spec_n)

        def seq(rt, t):
            # both pulses are driven through the resonator: the rotation angle of a
            # class scales with its coupling weight
            st = product_state(rt.M, [0.0, 0.0, -1.0], K_spec=rt.K)
            st = rotate_classes(st, Y, 0.5 * np.pi * ww, spec_angles=np.full(rt.K, 0.5 * np.pi))
            st = evolve(st, rt, t / 2, rtol=1e-6)
            st = rotate_classes(st, X, np.pi * ww, spec_angles=np.full(rt.K, np.pi))
            return evolve(st, rt, t / 2, rtol=1e-6)
        # unweighted collective spin (optical population readout)
        xi, t = opt_generic(p, ens3, seq, t_lo, t_hi, weights=None)
    elif kind == "noecho":
        xi, t = best_of(p, ens, False, t_lo, t_hi)
    elif kind == "ideal":
        xi, t = best_of(p, ens, True, t_lo, t_hi)
    else:
        raise ValueError(kind)
    return "c", label, kind, value, xi, t


def run(job):
    kind, a = job
    return {"a": job_a, "b": job_b, "c": job_c}[kind](a)


if __name__ == "__main__":
    out = {}
    fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    out["a_frac"] = fracs
    out["a_fid_1e_us"] = np.array([fid_1e_time(GAMMA_INH_HZ, f) for f in fracs]) * 1e6
    print("FID 1/e times (us) for 5 kHz:", dict(zip(fracs, np.round(out["a_fid_1e_us"], 1))), flush=True)
    # Lorentzian width that reproduces a 52 us exponential Ramsey decay: FWHM = 1/(pi T2*)
    out["a_lor_fwhm_for_52us_hz"] = 1.0 / (np.pi * 52e-6)
    T2s = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.15])
    out["b_T2"] = T2s
    jobs = [("a", (f, e)) for f in fracs for e in [True, False]]
    jobs += [("b", (T, e)) for T in T2s for e in [True, False]]
    cj = []
    for label, durs in [("lg", [1e-6, 3e-6, 1e-5]), ("sc", [1e-7, 3e-7, 1e-6])]:
        cj += [(label, "ideal", 0.0), (label, "noecho", 0.0)]
        cj += [(label, "duration", d) for d in durs]
        cj += [(label, "duration_noecho", d) for d in durs]
        cj += [(label, "angle", e) for e in [0.01, 0.03, 0.1]]
        cj += [(label, "spread", 0.02), (label, "spread", 0.05)]
    jobs += [("c", a) for a in cj]

    rows_a, rows_b, rows_c = [], [], []
    with ProcessPoolExecutor(2) as ex, Timer("robustness"):
        for r in ex.map(run, jobs):
            print(r, flush=True)
            if r[0] == "a":
                rows_a.append([r[1], 1 if r[2] else 0, r[3], r[4]])
            elif r[0] == "b":
                rows_b.append([r[1], 1 if r[2] else 0, r[3], r[4]])
            else:
                rows_c.append([0 if r[1] == "lg" else 1, ["ideal", "noecho", "duration", "duration_noecho", "angle", "spread"].index(r[2]), r[3], r[4], r[5]])
            out["a_rows"] = np.array(rows_a)
            out["b_rows"] = np.array(rows_b)
            out["c_rows"] = np.array(rows_c)
            save("robustness", **out)
