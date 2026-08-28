"""Benchmark against the experiment of Fukumori et al. (arXiv:2604.26909), Fig. 4:
gap-protected Ramsey coherence for chi N0/2pi = 0.1, 0.7, 2, 4, 7 kHz with the
dispersive loop-gap resonator (kappa/2pi = 660 kHz, Delta/2pi = 22 MHz, g/2pi = 15 mHz,
gamma_inh/2pi = 5 kHz).  Mean field for the three line shapes, plus the full
cumulant solution (contrast and hidden transverse variances) for the Voigt line.
"""
from common import *  # noqa
from cavsqueeze.protocols import ramsey_meanfield, ramsey_cumulant
from cavsqueeze import equal_probability_classes, lineshape

chiN_list = np.array([0.1e3, 0.7e3, 2e3, 4e3, 7e3])
times = np.linspace(0, 6e-3, 601)
out = dict(t=times, chiN=chiN_list)
p0 = from_hz(15e-3, 660e3, 22e6, T=0.08, T2=T2_SPIN)
for chiN_hz in chiN_list:
    N = TWO_PI * chiN_hz / p0.chi
    for shape in ["gaussian", "voigt", "lorentzian"]:
        # mean field is cheap: use a fine equal-probability grid (tails matter little for the contrast)
        ens = equal_probability_classes(lineshape(shape, TWO_PI * GAMMA_INH_HZ, LORENTZ_FRACTION), 2000, N)
        with Timer(f"MF chiN={chiN_hz:.0f} {shape}"):
            r = ramsey_meanfield(p0, ens, times)
        out[f"mf_{shape}_{int(chiN_hz)}"] = r["contrast"]
    # cumulant, Voigt, standard grid (Ramsey without echo)
    ens = standard_ensemble(N, TWO_PI * chiN_hz, "voigt", GRID_STD)
    tc = np.linspace(0, 6e-3, 121)
    with Timer(f"cumulant chiN={chiN_hz:.0f}"):
        rc = ramsey_cumulant(p0, ens, tc, rtol=1e-6)
    out[f"cum_t"] = tc
    out[f"cum_contrast_{int(chiN_hz)}"] = rc["contrast"]
    out[f"cum_xi2_{int(chiN_hz)}"] = rc["xi2"]
    out[f"cum_varmin_{int(chiN_hz)}"] = rc["var_min"] / (N / 4)
    out[f"cum_varmax_{int(chiN_hz)}"] = rc["var_max"] / (N / 4)
    out[f"N_{int(chiN_hz)}"] = N
    # 1/e times from mean field
    for shape in ["gaussian", "voigt", "lorentzian"]:
        c = out[f"mf_{shape}_{int(chiN_hz)}"]
        idx = np.where(c < np.exp(-1))[0]
        te = times[idx[0]] if len(idx) else np.inf
        print(f"chiN/2pi={chiN_hz/1e3:.1f} kHz {shape:10s} T_1/e = {te*1e6:.0f} us")
save("benchmark", **out)
