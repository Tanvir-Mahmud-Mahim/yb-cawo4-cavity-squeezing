"""Cost of reversing the resonator detuning for the twist-untwist readout.

The interaction-based (twist-untwist) readout needs the sign of the
cavity-mediated interaction to be reversed, which means moving the resonator
from +Delta to -Delta relative to the spins.  The eliminated model treats this
reversal as instantaneous and free.  In reality the resonator field cannot
follow a sudden change: the field that was in the resonator before the jump
rings down at rate kappa/2 at the new detuning and leaves the resonator.  That
field is proportional to the collective operator J-, so its loss acts on the
spins like collective emission.  Integrating the leaked photon flux gives
(4 g^2 / Delta^2) J+ J-, which equals the collective emission accumulated in a
time 4 / kappa at the dispersive rate Gamma_SR = g^2 kappa / Delta^2.  A slow
ramp instead passes through resonance, where the collective emission rate is
4 g^2 / kappa, and costs the equivalent of pi Delta tau_r / kappa of dispersive
emission for a linear ramp of duration tau_r; this is always larger than the
sudden-jump cost once tau_r exceeds 1 / kappa.

Part A verifies these statements with the exact spins-plus-resonator model
(N = 10, Dicke basis): the twist-untwist gain is computed with a sudden jump and
with linear ramps of several durations, and compared with the eliminated model
using (i) an ideal free reversal and (ii) the reversal followed by a fictitious
collective-emission interval of duration 4 / kappa (the emulation used at
large N).

Part B applies the validated emulation to the design points of the paper with
the cumulant solver: the twist-untwist gain versus detection noise at the
superconducting operating point (g sqrt N / 2 pi = 1 MHz, Delta / 2 pi = 30 MHz,
N = 1e10, 20 mK) for several resonator linewidths, and for the loop-gap device.
"""
from common import *  # noqa
import qutip as qt
from qutip import piqs
from cavsqueeze import cumulant as cu
from cavsqueeze.protocols import css_x, twist
from cavsqueeze.exact import xi2_from_moments
from concurrent.futures import ProcessPoolExecutor
import dataclasses


# ---------------------------------------------------------------------------
# Part A: exact model
# ---------------------------------------------------------------------------

class FullModel:
    def __init__(self, N, g, kappa, Delta, n_ph):
        self.N, self.g, self.kappa, self.Delta = N, g, kappa, Delta
        jx, jy, jz = piqs.jspin(N)
        jp, jm = piqs.jspin(N, "+"), piqs.jspin(N, "-")
        nds = jx.shape[0]
        a = qt.destroy(n_ph)
        Ic, Is = qt.qeye(n_ph), qt.qeye(nds)
        ens = piqs.Dicke(N=N)
        ens.hamiltonian = 0 * jz
        D_tls = qt.super_tensor(qt.liouvillian(0 * a.dag() * a), ens.liouvillian())
        self.H_int = g * (qt.tensor(a, jp) + qt.tensor(a.dag(), jm))
        self.H_det = qt.tensor(a.dag() * a, Is)
        self.c_ops = [np.sqrt(kappa) * qt.tensor(a, Is)]
        self.D_base = qt.liouvillian(self.H_int, self.c_ops) + D_tls
        self.D_det = qt.liouvillian(self.H_det)
        self.JX, self.JY, self.JZ = (qt.tensor(Ic, o) for o in (jx, jy, jz))
        self.rho0 = qt.tensor(qt.fock_dm(n_ph, 0), piqs.css(N, x=np.pi / 2, y=0.0, basis="dicke", coordinates="polar"))
        self.n_op = qt.tensor(a.dag() * a, Is)
        self.opts = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**8}

    def evolve(self, rho, t, Delta_fn=None, Delta=None):
        if t <= 0:
            return rho
        if Delta_fn is None:
            D = self.D_base + Delta * self.D_det
        else:
            D = qt.QobjEvo([self.D_base, [self.D_det, Delta_fn]])
        return qt.mesolve(D, rho, [0, t], options=self.opts).states[-1]

    def rotate_y(self, rho, phi):
        U = (-1j * phi * self.JY).expm()
        return U * rho * U.dag()

    def moments(self, rho):
        ops = [self.JX, self.JY, self.JZ]
        J = np.array([qt.expect(o, rho).real for o in ops])
        Cov = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Cov[i, j] = 0.5 * qt.expect(ops[i] * ops[j] + ops[j] * ops[i], rho).real - J[i] * J[j]
        return J, Cov


def gain_from(final_state_fn, N, eps=1e-3):
    Jp, _ = final_state_fn(+eps)
    Jm, _ = final_state_fn(-eps)
    J0, Cov0 = final_state_fn(0.0)
    dJ = (Jp - Jm) / (2 * eps)
    e3 = J0 / np.linalg.norm(J0)
    e = dJ - (dJ @ e3) * e3
    e = e / np.linalg.norm(e)
    slope = float(dJ @ e)
    var = float(e @ Cov0 @ e)
    sql = (N / 2) ** 2 / (N / 4)
    return slope**2 / var / sql, slope / (N / 2), var / (N / 4)


def exact_tu(fm: FullModel, t, reversal, tau_r=0.0):
    """Twist-untwist gain in the full model; reversal = 'jump' or 'ramp'."""
    D = fm.Delta

    def final(phi):
        rho = fm.evolve(fm.rho0, t, Delta=D)
        rho = fm.rotate_y(rho, phi)
        if reversal == "ramp" and tau_r > 0:
            rho = fm.evolve(rho, tau_r, Delta_fn=lambda tt: D * (1 - 2 * tt / tau_r))
        rho = fm.evolve(rho, t, Delta=-D)
        return fm.moments(rho)

    return gain_from(final, fm.N)


def eliminated_tu(N, g, kappa, Delta, t, emulate=False):
    """Twist-untwist gain in the eliminated model (exact Dicke-basis solution)."""
    chi = -4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
    Gsr = 4 * g**2 * kappa / (4 * Delta**2 + kappa**2)
    jx, jy, jz = piqs.jspin(N)
    jp, jm = piqs.jspin(N, "+"), piqs.jspin(N, "-")
    ens = piqs.Dicke(N=N)
    ens.collective_emission = Gsr
    ens.hamiltonian = chi * jp * jm
    Lp = ens.liouvillian()
    ens.hamiltonian = -chi * jp * jm
    Lm = ens.liouvillian()
    ens.hamiltonian = 0 * jz
    L0 = ens.liouvillian()
    rho0 = piqs.css(N, x=np.pi / 2, y=0.0, basis="dicke", coordinates="polar")
    opts = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**8}

    def final(phi):
        rho = qt.mesolve(Lp, rho0, [0, t], options=opts).states[-1]
        U = (-1j * phi * jy).expm()
        rho = U * rho * U.dag()
        if emulate:
            rho = qt.mesolve(L0, rho, [0, 4.0 / kappa], options=opts).states[-1]
        rho = qt.mesolve(Lm, rho, [0, t], options=opts).states[-1]
        ops = [jx, jy, jz]
        J = np.array([qt.expect(o, rho).real for o in ops])
        Cov = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Cov[i, j] = 0.5 * qt.expect(ops[i] * ops[j] + ops[j] * ops[i], rho).real - J[i] * J[j]
        return J, Cov

    return gain_from(final, N)


# ---------------------------------------------------------------------------
# Part B: cumulant model with the emulated reversal cost
# ---------------------------------------------------------------------------

def tu_with_reversal_cost(params, ens, t, sigma_det, weights=None, reversal_time=None, **kw):
    """Twist-untwist gain with the resonator ring-down emulated as a collective
    emission interval of duration 4/kappa (chi = 0) at the reversal."""
    rt = cu.Rates.from_params(params, ens)
    rt_neg = cu.Rates.from_params(params.with_flipped_detuning(), ens)
    rt_ring = cu.Rates.from_params(dataclasses.replace(params, Delta=params.Delta), ens)
    rt_ring.chi1 = 0.0
    rt_ring.delta = rt_ring.delta * 0.0  # detunings do not act during the fictitious interval
    rt_ring.gamma_phi = 0.0
    rt_ring.spec_delta = rt_ring.spec_delta * 0.0
    tau = 4.0 / params.kappa if reversal_time is None else reversal_time
    n = rt.n
    c = np.ones(rt.M) if weights is None else np.asarray(weights, float)

    def final(ph):
        st = twist(css_x(rt.M), rt, t, echo=True, **kw)
        st = cu.rotate(st, np.array([0.0, 1.0, 0.0]), ph)
        if tau > 0:
            st = cu.evolve(st, rt_ring, tau, **kw)
        st = twist(st, rt_neg, t, echo=True, **kw)
        return st

    phi = 1e-6
    st_p, st_m, st_0 = final(+phi), final(-phi), final(0.0)
    Jp = cu.collective_moments(st_p, n, weights, spec_n=rt.spec_n)[0]
    Jm = cu.collective_moments(st_m, n, weights, spec_n=rt.spec_n)[0]
    J0, Cov0, S1, S2 = cu.collective_moments(st_0, n, weights, spec_n=rt.spec_n)
    dJ = (Jp - Jm) / (2 * phi)
    e3 = J0 / np.linalg.norm(J0)
    e = dJ - (dJ @ e3) * e3
    e = e / np.linalg.norm(e)
    slope = float(dJ @ e)
    var = float(e @ Cov0 @ e)
    sig = np.atleast_1d(np.asarray(sigma_det, float))
    sql = (S1 / 2) ** 2 / (S2 / 4)
    return slope**2 / (var + sig**2) / sql


def job_B(args):
    label, N, p, tau = args
    ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
    t_list = np.geomspace(1e-5, 1e-3, 10)
    eps = np.geomspace(1e-9, 1e-1, 33)
    gains = np.array([tu_with_reversal_cost(p, ens, t, eps * N, reversal_time=tau, rtol=1e-6) for t in t_list])
    return label, tau, t_list, eps, gains


if __name__ == "__main__":
    out = {}
    # ---------------- Part A ----------------
    N = 10
    Delta = 1.0
    kappa = 0.1  # 2 Delta / kappa = 20
    r = 0.1
    g = r * Delta / np.sqrt(N)
    chi = 4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
    t = 1.0 / (chi * N)  # twist time (chi N t = 1)
    fm = FullModel(N, g, kappa, Delta, 6)
    rowsA = []
    with Timer("exact reversal"):
        g_free = eliminated_tu(N, g, kappa, Delta, t, emulate=False)
        g_emul = eliminated_tu(N, g, kappa, Delta, t, emulate=True)
        g_jump = exact_tu(fm, t, "jump")
        print(f"eliminated, free reversal: gain {dB(g_free[0]):.3f} dB (amp {g_free[1]:.3f}, var ratio {g_free[2]:.3f})", flush=True)
        print(f"eliminated + 4/kappa emission: gain {dB(g_emul[0]):.3f} dB (amp {g_emul[1]:.3f}, var ratio {g_emul[2]:.3f})", flush=True)
        print(f"full model, sudden jump:   gain {dB(g_jump[0]):.3f} dB (amp {g_jump[1]:.3f}, var ratio {g_jump[2]:.3f})", flush=True)
        rowsA.append([0.0, g_free[0], g_emul[0], g_jump[0]])
        for tau_k in [0.25, 1.0, 4.0, 16.0]:
            tau_r = tau_k / kappa
            g_ramp = exact_tu(fm, t, "ramp", tau_r)
            print(f"full model, ramp tau_r = {tau_k}/kappa: gain {dB(g_ramp[0]):.3f} dB (amp {g_ramp[1]:.3f}, var ratio {g_ramp[2]:.3f})", flush=True)
            rowsA.append([tau_k, np.nan, np.nan, g_ramp[0]])
    out["A_rows"] = np.array(rowsA)  # columns: kappa tau_r, gain free, gain emulated, gain full
    out["A_params"] = np.array([N, g, kappa, Delta, t])
    save("reversal", **out)

    # ---------------- Part B ----------------
    cases = []
    for kap in [3e3, 1e4, 3e4, 1e5]:
        p = from_hz(1e6 / np.sqrt(1e10), kap, 30e6, T=0.02, T2=T2_SPIN)
        cases.append((f"sc_k{int(kap)}", 1e10, p, None))
        cases.append((f"sc_k{int(kap)}_ideal", 1e10, p, 0.0))
    p_lg, N_lg = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    cases.append(("lg", N_lg, p_lg, None))
    cases.append(("lg_ideal", N_lg, p_lg, 0.0))
    with ProcessPoolExecutor(2) as ex, Timer("reversal design points"):
        for label, tau, t_list, eps, gains in ex.map(job_B, cases):
            out[f"B_{label}_t"] = t_list
            out[f"B_{label}_eps"] = eps
            out[f"B_{label}_gain"] = gains
            print(label, "tau", tau, "max gain (no detection noise)", dB(np.nanmax(gains[:, 0])), "dB", flush=True)
            save("reversal", **out)
