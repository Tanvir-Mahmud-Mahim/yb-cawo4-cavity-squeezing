"""Validation testbench data (Fig. 1 and Supplement):
 (a) N = 8 disordered spins: exact master equation versus cumulant.
 (b) Homogeneous N = 20, 40, 80 with collective decay and pumping: PIQS exact versus cumulant.
 (c) Homogeneous large-N limit: optimum squeezing versus 2 Delta/kappa against the
     (Gamma/chi)^(2/3) law of Lewis-Swan et al.
"""
from common import *  # noqa
from cavsqueeze.cumulant_raw import Rates as RatesRaw, product_state as ps_raw, evolve as ev_raw, wineland_xi2 as xi_raw, State as StRaw
from cavsqueeze.exact import full_hilbert, dicke_piqs, xi2_from_moments
from cavsqueeze.protocols import optimal_squeezing
from cavsqueeze import Rates, homogeneous

out = {}
# (a) disordered N = 8
rng = np.random.default_rng(7)
N = 8
delta = rng.normal(size=N) * 0.8
G = rng.uniform(0.7, 1.3, size=N)
rt = RatesRaw(delta=delta, G=G, n=np.ones(N), chi1=1.0 / N, Gd=0.05, Gu=0.01, gamma_phi=0.02)
times = np.linspace(0, 2.0, 41)
with Timer("exact N=8"):
    ex = full_hilbert(rt, [1, 0, 0], times)
sts = ev_raw(ps_raw(N, [1, 0, 0]), rt, times[-1], t_eval=times)
xi_c, xi_e, J_c, J_e = [], [], [], []
for k in range(len(times)):
    st_e = StRaw(ex["s"][k], ex["z"][k].astype(complex), ex["P"][k], ex["Q"][k], ex["R"][k], ex["Z"][k].astype(complex))
    xi_c.append(xi_raw(sts[k], rt.n)[0])
    xi_e.append(xi_raw(st_e, rt.n)[0])
    J_c.append(xi_raw(sts[k], rt.n)[4])
    J_e.append(xi_raw(st_e, rt.n)[4])
out.update(a_Q=times * 1.0, a_xi_cum=np.array(xi_c), a_xi_exact=np.array(xi_e), a_J_cum=np.array(J_c), a_J_exact=np.array(J_e),
           a_delta=delta, a_G=G)

# (b) PIQS convergence
bQ = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
b_res = {}
for Nb in [20, 40, 80]:
    chi = 1.0
    Gd, Gu, gphi = 0.6 * 20 / Nb, 0.15 * 20 / Nb, 0.05
    tb = np.concatenate([[0.0], bQ / (chi * Nb)])
    with Timer(f"PIQS N={Nb}"):
        ref = dicke_piqs(Nb, chi, Gd, Gu, gphi, np.pi / 2, tb)
    rtb = RatesRaw(delta=np.zeros(1), G=np.ones(1), n=np.array([Nb], float), chi1=chi, Gd=Gd, Gu=Gu, gamma_phi=gphi)
    stb = ev_raw(ps_raw(1, [1, 0, 0]), rtb, tb[-1], t_eval=tb)
    xc = [xi_raw(stb[k], rtb.n)[0] for k in range(1, len(tb))]
    xe = [xi2_from_moments(ref["J"][k], ref["Cov"][k], Nb) for k in range(1, len(tb))]
    out[f"b_xi_cum_N{Nb}"] = np.array(xc)
    out[f"b_xi_exact_N{Nb}"] = np.array(xe)
out["b_Q"] = bQ

# (c) Lewis-Swan scaling (homogeneous, no thermal photons, no T2)
ratios = np.array([10, 20, 50, 100, 200, 500, 1000, 3000, 10000], float)
xi_opt, t_opt, Q_opt, pred = [], [], [], []
for r in ratios:
    Nh, gN, kappa = 1e12, 1e6, 1e5
    Delta = r * kappa / 2
    p = from_hz(gN / np.sqrt(Nh), kappa, Delta, T=0.0, T2=None)
    ens = homogeneous(Nh)
    t_pred = (2 / (p.chi**2 * p.Gamma_SR * Nh**3)) ** (1 / 3)
    with Timer(f"LS ratio {r}"):
        best = optimal_squeezing(p, ens, t_pred / 10, t_pred * 10, echo=True, n_coarse=9)
    xi_opt.append(best["xi2"])
    t_opt.append(best["t"])
    Q_opt.append(best["Q"])
    pred.append(3 / 2 ** (2 / 3) * (p.Gamma_SR / p.chi) ** (2 / 3))
    print(r, dB(best["xi2"]), dB(pred[-1]), best["Q"])
out.update(c_ratio=ratios, c_xi_opt=np.array(xi_opt), c_t_opt=np.array(t_opt), c_Q_opt=np.array(Q_opt), c_lewis_swan=np.array(pred))
# fitted prefactor  xi2 = A (kappa/2Delta)^(2/3)
A = np.exp(np.mean(np.log(np.array(xi_opt)[3:]) - (2 / 3) * np.log(1 / ratios[3:])))
B = np.exp(np.mean(np.log(np.array(Q_opt)[3:]) - (1 / 3) * np.log(ratios[3:])))
out.update(c_prefactor_xi=A, c_prefactor_Q=B)
print("fitted prefactors: xi2 = %.3f (kappa/2Delta)^(2/3), Q_opt = %.3f (2Delta/kappa)^(1/3)" % (A, B))
save("validation", **out)
