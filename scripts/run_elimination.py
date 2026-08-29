"""Validity of the resonator elimination and of the rotating-wave approximation.

Part A (elimination).  A homogeneous ensemble of N spins coupled to one resonator
mode is solved exactly, with the mode kept as a quantum degree of freedom, in the
permutation-symmetric (Dicke) basis with PIQS:

    H = Delta a^dag a + g (a J+ + a^dag J-),   L = sqrt(kappa) a,

in the frame rotating at the spin frequency (Delta = omega_c - omega_s, T = 0).
The result is compared with the eliminated model used throughout the paper,

    H_eff = chi J+ J-,   L_eff = sqrt(Gamma_SR) J-,
    chi = -4 g^2 Delta / (4 Delta^2 + kappa^2),  Gamma_SR = 4 g^2 kappa / (4 Delta^2 + kappa^2),

solved exactly in the same basis.  (The sign of chi follows from second-order
perturbation theory for a resonator above the spin frequency; it does not affect
the squeezing parameter.)  The comparison is made at fixed 2 Delta / kappa for
several values of the elimination parameter g sqrt(N) / Delta, and it quantifies
what the design rule Delta >= 5 g sqrt(N) of the main text costs in accuracy.

Part B (rotating-wave approximation).  The same system is solved in the
laboratory frame without the rotating-wave approximation,

    H = omega_c a^dag a + omega_s Jz + g (a + a^dag)(J+ + J-),

for omega_s / Delta = 247 (the ratio of the superconducting design point,
3.08 GHz / 12.5 MHz) and for 140 (the loop-gap device, 3.08 GHz / 22 MHz), and
the squeezing parameter is compared with the rotating-wave result.  The
squeezing parameter and the contrast are invariant under rotations about z,
so no frame transformation is needed for the comparison.
"""
from common import *  # noqa
import qutip as qt
from qutip import piqs
from cavsqueeze.exact import dicke_piqs, xi2_from_moments


def _spin_ops(N):
    jx, jy, jz = piqs.jspin(N)
    jp = piqs.jspin(N, "+")
    jm = piqs.jspin(N, "-")
    return jx, jy, jz, jp, jm


def _moments(res, J_ops_idx):
    E = [np.real(e) for e in res.expect]
    J = np.stack(E[:3], axis=1)
    Cov = np.zeros((len(E[0]), 3, 3))
    Cov[:, 0, 0] = E[3] - J[:, 0] ** 2
    Cov[:, 1, 1] = E[4] - J[:, 1] ** 2
    Cov[:, 2, 2] = E[5] - J[:, 2] ** 2
    Cov[:, 0, 1] = Cov[:, 1, 0] = 0.5 * E[6] - J[:, 0] * J[:, 1]
    Cov[:, 0, 2] = Cov[:, 2, 0] = 0.5 * E[7] - J[:, 0] * J[:, 2]
    Cov[:, 1, 2] = Cov[:, 2, 1] = 0.5 * E[8] - J[:, 1] * J[:, 2]
    return J, Cov


def full_model(N, g, kappa, Delta, times, n_ph, rwa=True, omega_s=None, omega_c=None):
    """Exact spins + resonator (Dicke basis).  Returns (J, Cov, <a^dag a>)."""
    jx, jy, jz, jp, jm = _spin_ops(N)
    nds = jx.shape[0]
    a = qt.destroy(n_ph)
    Ic, Is = qt.qeye(n_ph), qt.qeye(nds)
    ens = piqs.Dicke(N=N)
    ens.hamiltonian = 0 * jz
    D_tls = ens.liouvillian()  # spin part (no local dissipation): only the basis structure
    if rwa:
        H = Delta * qt.tensor(a.dag() * a, Is) + g * (qt.tensor(a, jp) + qt.tensor(a.dag(), jm))
    else:
        H = omega_c * qt.tensor(a.dag() * a, Is) + omega_s * qt.tensor(Ic, jz) + g * qt.tensor(a + a.dag(), jp + jm)
    D_int = qt.liouvillian(H, [np.sqrt(kappa) * qt.tensor(a, Is)])
    D = D_int + qt.super_tensor(qt.liouvillian(0 * a.dag() * a), D_tls)
    rho0 = qt.tensor(qt.fock_dm(n_ph, 0), piqs.css(N, x=np.pi / 2, y=0.0, basis="dicke", coordinates="polar"))
    JX, JY, JZ = (qt.tensor(Ic, o) for o in (jx, jy, jz))
    ops = [JX, JY, JZ, JX * JX, JY * JY, JZ * JZ, JX * JY + JY * JX, JX * JZ + JZ * JX, JY * JZ + JZ * JY,
           qt.tensor(a.dag() * a, Is)]
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**8}
    if not rwa:
        opts["max_step"] = 0.05 / omega_c
    res = qt.mesolve(D, rho0, times, e_ops=ops, options=opts)
    J, Cov = _moments(res, None)
    return J, Cov, np.real(res.expect[9])


def eliminated(N, g, kappa, Delta, times):
    chi = -4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
    Gsr = 4 * g**2 * kappa / (4 * Delta**2 + kappa**2)
    r = dicke_piqs(N, chi, Gsr, 0.0, 0.0, np.pi / 2, times)
    return r["J"], r["Cov"], chi, Gsr


def xi2_trace(J, Cov, N):
    return np.array([xi2_from_moments(J[k], Cov[k], N) for k in range(len(J))])


if __name__ == "__main__":
    out = {}
    # ---------------- Part A: elimination ----------------
    N = 10
    two_delta_over_kappa = 20.0
    ratios = np.array([0.4, 0.2, 0.1, 0.05])  # g sqrt(N) / Delta
    Delta = 1.0
    kappa = 2 * Delta / two_delta_over_kappa
    rows = []
    with Timer("elimination"):
        for r in ratios:
            g = r * Delta / np.sqrt(N)
            chi = 4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
            t_max = 3.0 / (chi * N)
            times = np.linspace(0, t_max, 241)
            n_ph = 10 if r >= 0.4 else (7 if r >= 0.2 else 5)
            Jf, Cf, nph = full_model(N, g, kappa, Delta, times, n_ph)
            Je, Ce, chi_e, G_e = eliminated(N, g, kappa, Delta, times)
            xf, xe = xi2_trace(Jf, Cf, N), xi2_trace(Je, Ce, N)
            kf, ke = int(np.argmin(xf)), int(np.argmin(xe))
            cf = 2 * np.hypot(Jf[:, 0], Jf[:, 1]) / N
            ce = 2 * np.hypot(Je[:, 0], Je[:, 1]) / N
            rows.append([r, xf[kf], xe[ke], times[kf] * chi * N, times[ke] * chi * N, nph.max(), np.max(np.abs(xf - xe) / xe)])
            print(f"ratio {r}: full {dB(xf[kf]):.3f} dB at chiNt={times[kf]*chi*N:.3f}; eliminated {dB(xe[ke]):.3f} dB at {times[ke]*chi*N:.3f};"
                  f" max photons {nph.max():.3g}; max rel dev xi2 {rows[-1][-1]:.3g}", flush=True)
            out[f"A_r{r}_chiNt"] = times * chi * N
            out[f"A_r{r}_xi2_full"] = xf
            out[f"A_r{r}_xi2_elim"] = xe
            out[f"A_r{r}_con_full"] = cf
            out[f"A_r{r}_con_elim"] = ce
            out[f"A_r{r}_nph"] = nph
    out["A_rows"] = np.array(rows)
    out["A_N"] = N
    out["A_two_delta_over_kappa"] = two_delta_over_kappa
    save("elimination", **out)
    # N dependence at fixed g sqrt(N) / Delta = 0.2
    with Timer("elimination N = 20"):
        N20, r = 20, 0.2
        g = r * Delta / np.sqrt(N20)
        chi = 4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
        times = np.linspace(0, 2.5 / (chi * N20), 101)
        Jf, Cf, nph = full_model(N20, g, kappa, Delta, times, 7)
        Je, Ce, _, _ = eliminated(N20, g, kappa, Delta, times)
        xf, xe = xi2_trace(Jf, Cf, N20), xi2_trace(Je, Ce, N20)
        print(f"N = 20, ratio {r}: full {dB(xf.min()):.3f} dB; eliminated {dB(xe.min()):.3f} dB; max rel dev {np.max(np.abs(xf - xe) / xe):.3g}", flush=True)
        out["A20_chiNt"], out["A20_xi2_full"], out["A20_xi2_elim"] = times * chi * N20, xf, xe
    save("elimination", **out)

    # ---------------- Part B: rotating-wave approximation ----------------
    N = 4
    r = 0.2
    rowsB = []
    with Timer("rwa"):
        for label, ws_over_delta in [("sc", 247.0), ("lg", 140.0)]:
            Delta = 1.0
            kappa = 2 * Delta / two_delta_over_kappa
            g = r * Delta / np.sqrt(N)
            chi = 4 * g**2 * Delta / (4 * Delta**2 + kappa**2)
            t_max = 2.0 / (chi * N)
            times = np.linspace(0, t_max, 161)
            omega_s = ws_over_delta * Delta
            omega_c = omega_s + Delta
            Jr, Cr, _ = full_model(N, g, kappa, Delta, times, 6, rwa=True)
            Jl, Cl, _ = full_model(N, g, kappa, Delta, times, 6, rwa=False, omega_s=omega_s, omega_c=omega_c)
            xr, xl = xi2_trace(Jr, Cr, N), xi2_trace(Jl, Cl, N)
            kr, kl = int(np.argmin(xr)), int(np.argmin(xl))
            dev = np.max(np.abs(xl - xr) / xr)
            rowsB.append([ws_over_delta, xr[kr], xl[kl], times[kr] * chi * N, times[kl] * chi * N, dev])
            print(f"{label}: RWA {dB(xr[kr]):.4f} dB at {times[kr]*chi*N:.3f}; lab frame {dB(xl[kl]):.4f} dB at {times[kl]*chi*N:.3f};"
                  f" max rel dev {dev:.3g}; expected order Delta/(omega_c+omega_s) = {Delta/(omega_c+omega_s):.2e}", flush=True)
            out[f"B_{label}_chiNt"] = times * chi * N
            out[f"B_{label}_xi2_rwa"] = xr
            out[f"B_{label}_xi2_lab"] = xl
    out["B_rows"] = np.array(rowsB)
    out["B_N"] = N
    out["B_ratio"] = r
    save("elimination", **out)
