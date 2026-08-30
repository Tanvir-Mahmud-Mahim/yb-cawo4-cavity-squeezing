"""Spin echo against the cavity-mediated interaction (prediction for the demonstrated device).

A Hahn echo (pi/2, wait tau, pi, wait tau) refocuses the static spread of
transition frequencies: the pi pulse reverses the sign of the detuning term of
the Hamiltonian.  It does not reverse the exchange term chi J+ J-, which is
even under the rotation, so the phases from the exchange field are repeated
rather than undone in the second half of the sequence.  When the
interaction chi N is much larger than the line width the ensemble is locked and
little dephasing happens in the first place; when it is much smaller the
dephasing is static and the echo works.  In between, the echo is predicted to
collapse.  The scan below follows the echo contrast of the loop-gap device as
the pumped spin number N0 (and with it chi N0) is varied, exactly as in the
experiment, at several echo times, for the three line shapes; the same
quantity is checked with the cumulant solver (an exact few-spin check is not
possible here: for N <= 10 the one-axis-twisting part of the same interaction
collapses the contrast within sqrt(N)/(chi N), comparable to the echo times of
interest; for 1e14 to 2e15 spins that time is 6 to 26 minutes).  Outputs: data/echo.npz.
"""
from common import *  # noqa
from cavsqueeze.cumulant import Rates, evolve, evolve_meanfield, collective_moments, rotate, coherence
from cavsqueeze.protocols import css_x
from cavsqueeze import equal_probability_classes, lineshape
from concurrent.futures import ProcessPoolExecutor

N0S = np.geomspace(1e13, 2e15, 16)
TAUS = np.array([1e-4, 3e-4, 1e-3, 3e-3])
M_MF = 500


def mf_echo(p, N, tau, frac, M=M_MF, echo=True):
    """Mean-field contrast at 2 tau with (echo=True) or without a pi pulse at tau, for a
    Voigt line of Lorentzian fraction frac (0: Gaussian, 1: Lorentzian)."""
    ens = equal_probability_classes(lineshape("voigt", TWO_PI * GAMMA_INH_HZ, frac), M, N)
    rt = Rates.from_params(p, ens)
    s0 = np.full(rt.M, 0.5, complex)
    z0 = np.zeros(rt.M, complex)
    s1, z1 = evolve_meanfield(s0, z0, rt, tau, rtol=1e-7, atol=1e-10)
    if echo:
        s1, z1 = np.conj(s1), -z1
    s2, z2 = evolve_meanfield(s1, z1, rt, tau, rtol=1e-7, atol=1e-10)
    return 2 * abs(rt.n @ s2) / N


def job_mf(args):
    N0, frac = args
    p, N = loop_gap_dispersive(N0, T=0.08, T2=T2_SPIN)
    return [mf_echo(p, N, tau, frac) for tau in TAUS] + [mf_echo(p, N, tau, frac, echo=False) for tau in TAUS]


def job_cumulant(args):
    N0, tau = args
    p, N = loop_gap_dispersive(N0, T=0.08, T2=T2_SPIN)
    ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
    rt = Rates.from_params(p, ens)
    st = evolve(css_x(rt.M), rt, tau, rtol=1e-7)
    st = rotate(st, [1.0, 0.0, 0.0], np.pi)
    st = evolve(st, rt, tau, rtol=1e-7)
    return coherence(st, rt.n, spec_n=rt.spec_n)


if __name__ == "__main__":
    out = dict(N0s=N0S, taus=TAUS)
    prev = load("echo")
    if prev is not None:  # keep the results of run_echo_extra.py, if any
        out.update({k: v for k, v in prev.items() if k.startswith(("fine_", "noem_", "conv_", "tau_fine"))})
    shapes = [(0.3, "voigt"), (0.0, "gaussian"), (1.0, "lorentzian")]
    if "--tail" in sys.argv:  # keep the stored scan, redo only the checks
        out.update(prev)
        shapes = []
    with ProcessPoolExecutor(3) as ex, Timer("mean-field echo scan"):
        for frac, tag in shapes:
            rows = np.array(list(ex.map(job_mf, [(N0, frac) for N0 in N0S])))
            out[f"mf_{tag}_echo"] = rows[:, : len(TAUS)]
            out[f"mf_{tag}_ramsey"] = rows[:, len(TAUS):]
            p0, _ = loop_gap_dispersive(1.0)
            for N0, r in zip(N0S, rows):
                print(f"{tag}: N0={N0:.2e} chiN0/2pi={p0.chi*N0/TWO_PI:8.0f} Hz  echo C(2tau)={np.round(r[:len(TAUS)],3)}  ramsey={np.round(r[len(TAUS):],3)}", flush=True)
            save("echo", **out)
    # cumulant check (light grid) at tau = 1 ms for a subset of N0
    with ProcessPoolExecutor(3) as ex, Timer("cumulant echo check"):
        sub = N0S[::3]
        vals = list(ex.map(job_cumulant, [(N0, 1e-3) for N0 in sub]))
        out["cum_N0s"] = sub
        out["cum_echo_1ms"] = np.array(vals)
        for N0, v in zip(sub, vals):
            print(f"cumulant: N0={N0:.2e} echo C(2 ms)={v:.3f}", flush=True)
        save("echo", **out)
    save("echo", **out)
