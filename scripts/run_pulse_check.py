"""Exact check of the finite-pulse treatment.

A homogeneous ensemble of N = 60 spins under H = chi J+ J- (no dissipation) is
twisted for a total time Q = chi N t = 1 with a pi pulse about x at t/2.  The pi
pulse is either instantaneous or has a finite duration tau_p during which the
drive Omega Jx (Omega = pi / tau_p) is added to the interaction, and the
squeezing parameter is computed exactly in the Dicke basis.  The same sequence
is computed with the cumulant solver and its Strang-split pulse (protocols.pulse)
for N = 60 and for N = 1e10, and the convergence of the splitting with the
number of increments is recorded.  Results -> data/pulse_check.json.
"""
from common import *  # noqa
import qutip as qt
from qutip import piqs
from cavsqueeze.exact import xi2_from_moments
from cavsqueeze.cumulant import Rates, evolve, rotate, wineland_xi2
from cavsqueeze.protocols import pulse, css_x, X

if __name__ == "__main__":
    N, chi, Q = 60, 1.0, 1.0
    jx, jy, jz = piqs.jspin(N)
    jp = piqs.jspin(N, "+")
    H = chi * jp * jp.dag()
    rho0 = piqs.css(N, x=np.pi / 2, y=0.0, basis="dicke", coordinates="polar")
    t = Q / (chi * N)
    opts = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**7}

    def moments(rho):
        ops = [jx, jy, jz]
        J = np.array([qt.expect(o, rho).real for o in ops])
        C = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                C[i, j] = 0.5 * qt.expect(ops[i] * ops[j] + ops[j] * ops[i], rho).real - J[i] * J[j]
        return J, C

    def exact(tau):
        rho = qt.mesolve(H, rho0, [0, t / 2], options=opts).states[-1]
        if tau == 0:
            U = (-1j * np.pi * jx).expm()
            rho = U * rho * U.dag()
        else:
            rho = qt.mesolve(H + (np.pi / tau) * jx, rho, [0, tau], options=opts).states[-1]
        rho = qt.mesolve(H, rho, [0, t / 2], options=opts).states[-1]
        J, C = moments(rho)
        return xi2_from_moments(J, C, N)

    out = {"N_exact": N, "Q": Q, "chiN_tau": [0.05, 0.1, 0.2]}
    with Timer("exact N = 60"):
        x0 = exact(0.0)
        out["exact_ideal"] = x0
        out["exact_rel_increase"] = [exact(q / (chi * N)) / x0 - 1 for q in out["chiN_tau"]]
        print("exact N=60: relative increase of xi^2:", np.round(out["exact_rel_increase"], 4), flush=True)
    for NN, tag in [(60, "cumulant_N60"), (1e10, "cumulant_N1e10")]:
        p = from_hz(1e6 / np.sqrt(NN), 1e4, 30e6, T=0.0, T2=None)
        rt = Rates.from_params(p, homogeneous(NN))
        rt.Gd = 0.0
        rt.Gu = 0.0
        chiN = rt.chiN
        tt = Q / chiN
        st = evolve(css_x(1), rt, tt / 2, rtol=1e-9)
        st = rotate(st, X, np.pi)
        st = evolve(st, rt, tt / 2, rtol=1e-9)
        x0 = wineland_xi2(st, rt.n)[0]
        rel = []
        for q in out["chiN_tau"]:
            st = evolve(css_x(1), rt, tt / 2, rtol=1e-9)
            st = pulse(st, rt, X, np.pi, duration=q / chiN, n_steps=40, rtol=1e-9)
            st = evolve(st, rt, tt / 2, rtol=1e-9)
            rel.append(wineland_xi2(st, rt.n)[0] / x0 - 1)
        out[tag + "_ideal"] = x0
        out[tag + "_rel_increase"] = rel
        print(tag, "relative increase of xi^2:", np.round(rel, 4), flush=True)
    # convergence of the splitting at the superconducting operating point, 1 us pulse
    p = from_hz(1e6 / np.sqrt(1e10), 1e4, 30e6, T=0.02, T2=T2_SPIN)
    rt = Rates.from_params(p, homogeneous(1e10))
    conv = {}
    for n in [10, 20, 40, 80]:
        st = evolve(css_x(1), rt, 2e-5, rtol=1e-9)
        st = pulse(st, rt, X, np.pi, duration=1e-6, n_steps=n, rtol=1e-9)
        st = evolve(st, rt, 2e-5, rtol=1e-9)
        conv[n] = wineland_xi2(st, rt.n)[0]
    out["splitting_convergence"] = conv
    out["splitting_rel_change_20_to_80"] = conv[20] / conv[80] - 1
    print("splitting convergence:", conv, flush=True)
    save_json("pulse_check", out)
