"""Echo prediction, additional scans.

(1) Fine linear scan of the echo half-time tau at the operating point (N0 = 6e14 and 7e14,
    the two inferences of the pumped spin number), with and without the collective emission:
    the oscillation at the collective precession frequency and its damping.
(2) The N0 scan at tau = 0.1 and 0.3 ms with the collective emission switched off (echo and
    no pulse), to separate the exchange interaction from superradiance.
(3) Convergence of the mean-field class number at the most sensitive point.
Outputs are added to data/echo.npz.
"""
from common import *  # noqa
from run_echo import mf_echo, N0S
from cavsqueeze.cumulant import Rates, evolve_meanfield
from cavsqueeze import equal_probability_classes, lineshape
from concurrent.futures import ProcessPoolExecutor

TAU_FINE = np.arange(1e-5, 6.0e-4 + 1e-9, 5e-6)


def mf_contrast(p, N, tau, frac=0.3, M=500, echo=True, emission=True):
    ens = equal_probability_classes(lineshape("voigt", TWO_PI * GAMMA_INH_HZ, frac), M, N)
    rt = Rates.from_params(p, ens)
    if not emission:
        rt.Gd = 0.0
        rt.Gu = 0.0
    s0 = np.full(rt.M, 0.5, complex)
    z0 = np.zeros(rt.M, complex)
    s1, z1 = evolve_meanfield(s0, z0, rt, tau, rtol=1e-7, atol=1e-10)
    if echo:
        s1, z1 = np.conj(s1), -z1
    s2, z2 = evolve_meanfield(s1, z1, rt, tau, rtol=1e-7, atol=1e-10)
    return 2 * abs(rt.n @ s2) / N


def job_fine(args):
    N0, tau = args
    p, N = loop_gap_dispersive(N0, T=0.08, T2=T2_SPIN)
    return [mf_contrast(p, N, tau, echo=e, emission=m) for e in (True, False) for m in (True, False)]


def job_noem(args):
    N0, tau = args
    p, N = loop_gap_dispersive(N0, T=0.08, T2=T2_SPIN)
    return [mf_contrast(p, N, tau, echo=True, emission=False), mf_contrast(p, N, tau, echo=False, emission=False)]


def job_conv(M):
    p, N = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    return mf_contrast(p, N, 3e-4, M=M)


if __name__ == "__main__":
    out = dict(load("echo"))
    with ProcessPoolExecutor(3) as ex, Timer("fine tau scans"):
        for N0 in [6e14, 7e14]:
            rows = np.array(list(ex.map(job_fine, [(N0, tau) for tau in TAU_FINE])))
            out[f"fine_N0_{N0:.0e}"] = rows  # columns: echo, echo no emission, no pulse, no pulse no emission
            print(f"N0={N0:.0e}: echo min {rows[:,0].min():.3f} at {TAU_FINE[np.argmin(rows[:,0])]*1e6:.0f} us; "
                  f"echo at 0.1 ms {np.interp(1e-4, TAU_FINE, rows[:,0]):.3f} (no emission {np.interp(1e-4, TAU_FINE, rows[:,1]):.3f}); "
                  f"no pulse at 0.1 ms {np.interp(1e-4, TAU_FINE, rows[:,2]):.3f}", flush=True)
        out["tau_fine"] = TAU_FINE
        save("echo", **out)
    with ProcessPoolExecutor(3) as ex, Timer("no-emission N0 scans"):
        for tau, tag in [(1e-4, "01"), (3e-4, "03")]:
            rows = np.array(list(ex.map(job_noem, [(N0, tau) for N0 in N0S])))
            out[f"noem_{tag}"] = rows  # columns: echo, no pulse
            print(f"no emission, tau={tau*1e3:g} ms: echo {np.round(rows[:,0],3)}", flush=True)
        save("echo", **out)
    with ProcessPoolExecutor(3) as ex, Timer("class convergence"):
        Ms = [250, 500, 1000, 2000]
        vals = list(ex.map(job_conv, Ms))
        out["conv_M"] = np.array(Ms)
        out["conv_echo"] = np.array(vals)
        print("convergence at N0=6e14, tau=0.3 ms:", dict(zip(Ms, np.round(vals, 4))), flush=True)
    save("echo", **out)
