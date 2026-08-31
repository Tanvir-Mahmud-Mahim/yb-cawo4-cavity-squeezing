"""Discretization convergence tables for the Supplement (Tables S1, S2)."""
from common import *  # noqa
from cavsqueeze.protocols import squeezing_after
from cavsqueeze import equal_probability_classes

fw = TWO_PI * GAMMA_INH_HZ
d = lineshape("voigt", fw, LORENTZ_FRACTION)
rows = []
N = 1e10
p = from_hz(1e6 / np.sqrt(N), 1e4, 30e6, T=0.02, T2=T2_SPIN)
p2, N2 = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
cases = [("SC", p, N, 0.2e-3, True), ("LG-echo", p2, N2, 0.364e-3, True), ("LG-free", p2, N2, 0.16e-3, False)]
for label, pp, NN, t, echo in cases:
    chiN = pp.chi * NN
    for (Mc, Mt, fac) in [(16, 10, 5), (24, 12, 5), (32, 12, 5), (24, 16, 5), (48, 16, 5), (48, 16, 10), (64, 24, 5), (96, 16, 5)]:
        ens = tail_resolved_classes(d, NN, Mc, Mt, fwhm=fw, core_edge=3 * fw, delta_max=1000 * fw, spectator_beyond=fac * max(chiN, 2 * fw))
        with Timer(f"{label} {Mc} {Mt} {fac}"):
            r = squeezing_after(pp, ens, t, echo=echo, rtol=1e-6)
        rows.append([label, Mc, Mt, fac, ens.M, len(ens.spec_delta), float(dB(r["xi2"])), float(r["contrast"])])
        print(rows[-1], flush=True)
    for M in [16, 32, 64]:
        ens = equal_probability_classes(d, M, NN)
        r = squeezing_after(pp, ens, t, echo=echo, rtol=1e-6)
        rows.append([label + "-equalprob", M, 0, 0, M, 0, float(dB(r["xi2"])), float(r["contrast"])])
        print(rows[-1], flush=True)
save_json("convergence", rows)
