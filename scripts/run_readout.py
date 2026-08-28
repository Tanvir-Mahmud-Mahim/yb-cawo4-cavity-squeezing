"""Readout requirements: metrological gain versus detection noise for
 (i) the plain squeezed-state readout and (ii) the twist-untwist (interaction-based)
readout, for the loop-gap device (N0 = 6e14, Delta/2pi = 22 MHz) and for
superconducting resonators with N = 1e9, 1e10, 1e11 (kappa/2pi = 10 kHz, g sqrt N = 1 MHz,
Delta/2pi = 30 MHz, 20 mK).  Detection noise is given as a fraction of N
(population resolution).  The interaction time is optimised for each detection noise.
"""
from common import *  # noqa
from cavsqueeze.protocols import twist_untwist, plain_squeezed_readout
from concurrent.futures import ProcessPoolExecutor

eps = np.geomspace(1e-9, 1e-1, 33)  # sigma_det / N
t_list = np.geomspace(1e-5, 2e-3, 12)


def job(args):
    label, N, p = args
    ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
    sig = eps * N
    gain_tu = np.full((len(t_list), len(eps)), np.nan)
    gain_pl = np.full((len(t_list), len(eps)), np.nan)
    amp = np.zeros(len(t_list))
    for k, t in enumerate(t_list):
        tu = twist_untwist(p, ens, t, sigma_det=sig, rtol=1e-6)
        pl = plain_squeezed_readout(p, ens, t, sigma_det=sig, rtol=1e-6)
        gain_tu[k] = tu["gain"]
        gain_pl[k] = pl["gain"]
        amp[k] = tu["amplification"]
    return label, N, gain_tu, gain_pl, amp


if __name__ == "__main__":
    cases = []
    p_lg, N_lg = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    cases.append(("loopgap", N_lg, p_lg))
    for N in [1e9, 1e10, 1e11]:
        cases.append((f"sc_{int(np.log10(N))}", N, from_hz(1e6 / np.sqrt(N), 1e4, 30e6, T=0.02, T2=T2_SPIN)))
    out = dict(eps=eps, t=t_list)
    with ProcessPoolExecutor(2) as ex, Timer("readout"):
        for label, N, gtu, gpl, amp in ex.map(job, cases):
            print(label, "max TU gain", dB(np.nanmax(gtu[:, 0])), "max plain gain", dB(np.nanmax(gpl[:, 0])), flush=True)
            out[f"{label}_N"] = N
            out[f"{label}_gain_tu"] = gtu
            out[f"{label}_gain_plain"] = gpl
            out[f"{label}_amp"] = amp
            save("readout", **out)
