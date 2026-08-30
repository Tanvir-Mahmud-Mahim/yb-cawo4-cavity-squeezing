"""Measurement-based squeezing through the resonator (Sec. "Measure instead of twist").

The resonator, probed at its own frequency, measures J_z of the ensemble
without disturbing it (a quantum non-demolition measurement): every spin shifts
the resonator frequency by +-g^2/Delta, so the phase of the reflected probe
carries 8 (g^2/Delta) J_z / kappa.  J_z commutes with the detunings of the
spins, so the measurement rate does not depend on the line shape (the line
enters only through the emission noise and the contrast below).  Reading
the phase with a quantum-limited amplifier of total efficiency eta for a time
t resolves J_z to a variance 1/(Gamma_m t) with

    Gamma_m = 64 eta (g^2/Delta)^2 n_bar / kappa      (one-port reflection),

n_bar being the mean intracavity photon number (Supplemental Material).
Conditioned on the record, the variance of J_z obeys

    dV/dt = - Gamma_m V^2 + D(t),

where D(t) = Gamma_SR [(n_th+1) <J+J-> + n_th <J-J+>] + N/(2 T1) is the rate at
which collective emission and absorption through the resonator, plus
independent spin flips at the population lifetime T1, randomise J_z
(<J+J-> = |<J_perp>|^2 + N/4 for a product state on the equator): the
radiation damping of the free-induction signal.  |J_perp(t)| and <J_z>(t) come from the mean-field solution with the
actual line and interaction, so D(t) knows whether the ensemble stays locked
(chi N > gamma_inh) or dephases.  The metrological squeezing is the Wineland
parameter 1/xi^2 = (N/4)/V x C(t)^2 with C the contrast, taken either directly
(locked ensemble) or after a spin echo (dephased ensemble; the echo leaves J_z
alone).  Outputs: data/measurement.npz.
"""
from common import *  # noqa
from cavsqueeze.cumulant import Rates, evolve, evolve_meanfield, collective_moments, rotate
from cavsqueeze.protocols import css_x, X
from cavsqueeze import equal_probability_classes, lineshape
from scipy.integrate import solve_ivp
from cavsqueeze.resonator import HBAR
from concurrent.futures import ProcessPoolExecutor

KAPPA_HZ = 1e4
GN_HZ = 1e6
T_BATH = 0.02
T_END = 0.1
T1_SPIN = 3 * 3600.0  # s; lower bound on the spin population lifetime at 50 mK (Tiranov et al. 2026, > 3 h recovery)
N_BARS = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
ETAS = [0.5, 0.8]
M_MF = 500


def meanfield_trajectory(p, N, times, shape="voigt", M=M_MF, echo_at=None):
    """Mean-field |J_perp|^2, <Jz>, <J+J->, <J-J+> and contrast versus time; with
    echo_at = tau a pi pulse about x is applied at tau."""
    ens = equal_probability_classes(lineshape(shape, TWO_PI * GAMMA_INH_HZ, LORENTZ_FRACTION), M, N)
    rt = Rates.from_params(p, ens)
    s0 = np.full(rt.M, 0.5, complex)
    z0 = np.zeros(rt.M, complex)
    if echo_at is None:
        s, z = evolve_meanfield(s0, z0, rt, float(times[-1]), t_eval=times, rtol=1e-7, atol=1e-10)
    else:
        t1 = times[times < echo_at]
        t2 = times[times >= echo_at]
        s1, z1 = evolve_meanfield(s0, z0, rt, float(echo_at), t_eval=np.append(t1, echo_at), rtol=1e-7, atol=1e-10)
        # pi pulse about x: s -> conj(s), z -> -z
        s_e, z_e = np.conj(s1[:, -1]), -z1[:, -1]
        s2, z2 = evolve_meanfield(s_e, z_e, rt, float(times[-1] - echo_at), t_eval=t2 - echo_at, rtol=1e-7, atol=1e-10)
        s = np.concatenate([s1[:, :-1], s2], axis=1)
        z = np.concatenate([z1[:, :-1], z2], axis=1)
    Jperp = rt.n @ s
    Jz = 0.5 * np.real(rt.n @ z)
    # product state: <J+J-> = |<J->|^2 + sum_i [(1+<z_i>)/2 - |<s_i>|^2]
    inc = np.abs(s) ** 2
    JpJm = np.abs(Jperp) ** 2 + np.real(rt.n @ (0.5 * (1 + z) - inc))
    JmJp = np.abs(Jperp) ** 2 + np.real(rt.n @ (0.5 * (1 - z) - inc))
    return dict(t=times, N=N, Jperp2=np.abs(Jperp) ** 2, Jz=Jz, JpJm=JpJm, JmJp=JmJp, contrast=2 * np.abs(Jperp) / N)


def gamma_m(p, n_bar, eta):
    chi_s = p.g**2 / p.Delta
    return 64.0 * eta * chi_s**2 * n_bar / p.kappa


def noise_rate(p, traj):
    """D(t): rate at which collective emission and absorption randomise J_z."""
    nth = p.n_th
    N = traj["N"]
    return p.Gamma_SR * ((nth + 1) * traj["JpJm"] + nth * traj["JmJp"]) + N / (2 * T1_SPIN)


def conditional_variance(p, N, traj, n_bar, eta, t_end, t_eval):
    """Conditional variance of J_z along the trajectory `traj` (free or with echo)."""
    Gm = gamma_m(p, n_bar, eta)
    D_t = noise_rate(p, traj)
    tt = traj["t"]

    def rhs(t, y):
        return [-Gm * y[0] ** 2 + np.interp(t, tt, D_t)]

    sol = solve_ivp(rhs, (0, t_end), [N / 4], t_eval=t_eval, rtol=1e-8, atol=1e-3, method="LSODA")
    return sol.y[0], Gm


TAUS = np.array([2.5e-4, 1e-3, 5e-3, 2.5e-2, 5e-2])
Z_MAX = 0.3  # echo results are kept only if the ensemble is still near the equator at the pi pulse


def job(args):
    """Direct protocol: measure from t=0 along the free trajectory, read at t.
    Echo protocol: measure from 0 to 2 tau with a pi pulse at tau, read at 2 tau;
    D(t) is taken along the echo trajectory itself (the emission of the refocusing
    ensemble is included).  Echo points with a strongly inverted ensemble at the
    pi pulse (|<J_z>| > Z_MAX N/2) are discarded: the mean-field emission noise is
    not trusted for an inverted ensemble."""
    N, Delta_hz = args
    p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, Delta_hz, T=T_BATH, T2=T2_SPIN)
    times = np.linspace(0, T_END, 4001)
    traj = meanfield_trajectory(p, N, times)
    t_eval = np.geomspace(1e-5, T_END, 121)
    echo_trajs = []
    for tau in TAUS:
        te = np.linspace(0, 2 * tau, 1601)
        echo_trajs.append(meanfield_trajectory(p, N, te, echo_at=tau, M=300))
    C_echo = np.array([tr["contrast"][-1] for tr in echo_trajs])
    z_pulse = np.array([2 * np.interp(tau, tr["t"], tr["Jz"]) / N for tau, tr in zip(TAUS, echo_trajs)])
    chi_s = p.g**2 / p.Delta
    n_crit = p.Delta**2 / (4 * p.g**2)          # dispersive approximation valid for n_bar << n_crit
    omega = p.omega_s + p.Delta
    res = dict(N=N, Delta=Delta_hz, chiN=p.chi * N, GN=p.Gamma_SR * N, t=times, contrast=traj["contrast"], Jz=traj["Jz"],
               D=noise_rate(p, traj), t_eval=t_eval, taus=TAUS, C_echo=C_echo, z_pulse=z_pulse, n_crit=n_crit,
               P_in_per_photon=p.kappa * HBAR * omega / 4,      # W per intracavity photon (one-port, on resonance)
               stark_per_photon=2 * chi_s,                      # rad/s spin frequency shift per photon
               linearity=2 * chi_s * np.sqrt(N) / p.kappa)      # sqrt(N) spin-flip shift over kappa
    for eta in ETAS:
        for nb in N_BARS:
            tag = f"eta{eta}_n{int(np.log10(nb))}"
            V, Gm = conditional_variance(p, N, traj, nb, eta, T_END, t_eval)
            S = (N / 4) / V
            W_direct = S * np.interp(t_eval, times, traj["contrast"]) ** 2  # Wineland gain 1/xi^2
            W_echo = np.empty(len(TAUS))
            for k, (tau, tr) in enumerate(zip(TAUS, echo_trajs)):
                Ve, _ = conditional_variance(p, N, tr, nb, eta, 2 * tau, np.array([2 * tau]))
                W_echo[k] = (N / 4) / Ve[0] * C_echo[k] ** 2
                if abs(z_pulse[k]) > Z_MAX:
                    W_echo[k] = np.nan
            if nb > 0.1 * n_crit:
                W_direct = np.full_like(W_direct, np.nan)
                W_echo = np.full_like(W_echo, np.nan)
            res[f"S_{tag}"] = S
            res[f"Wdirect_{tag}"] = W_direct
            res[f"Wecho_{tag}"] = W_echo
            res[f"Gm_{tag}"] = Gm
    return res


if __name__ == "__main__":
    Ns = [1e9, 1e10, 1e11]
    Deltas = [1e7, 3e7, 1e8, 2e8, 3e8, 1e9]
    jobs = [(N, D) for N in Ns for D in Deltas]
    out = dict(N_bars=N_BARS, etas=np.array(ETAS), Ns=np.array(Ns), Deltas=np.array(Deltas))
    if "--tail" in sys.argv:  # redo only the checks below, keeping the stored map
        out.update(load("measurement"))
        jobs = []
    with ProcessPoolExecutor(2) as ex, Timer("measurement map"):
        for res in ex.map(job, jobs):
            tag = f"N{int(np.log10(res['N']))}_D{int(res['Delta']/1e6)}"
            for k, v in res.items():
                out[f"{tag}_{k}"] = v
            def best(nb):
                wd = res[f"Wdirect_eta0.5_n{int(np.log10(nb))}"]
                we = res[f"Wecho_eta0.5_n{int(np.log10(nb))}"]
                d = np.nanmax(wd) if np.any(np.isfinite(wd)) else np.nan
                e = np.nanmax(we) if np.any(np.isfinite(we)) else np.nan
                return f"n{int(np.log10(nb))}:{dB(d):.1f}/{dB(e):.1f}"
            line = " ".join(best(nb) for nb in N_BARS)
            print(f"N={res['N']:.0e} Delta={res['Delta']/1e6:.0f} MHz chiN/2pi={res['chiN']/TWO_PI:.0f} Hz GN/2pi={res['GN']/TWO_PI:.3g} Hz "
                  f"C(1ms)={np.interp(1e-3,res['t'],res['contrast']):.2f} C_echo={np.round(res['C_echo'],2)} z_pulse={np.round(res['z_pulse'],2)}"
                  f" | best 1/xi^2 direct/echo (eta=0.5, dB): {line}", flush=True)
            save("measurement", **out)
    # loop-gap device
    p_lg, N_lg = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    times = np.linspace(0, 1e-2, 2001)
    traj = meanfield_trajectory(p_lg, N_lg, times)
    t_eval = np.geomspace(1e-5, 1e-2, 61)
    for nb in [1e8, 1e10, 1e12]:
        V, Gm = conditional_variance(p_lg, N_lg, traj, nb, 0.5, 1e-2, t_eval)
        S = (N_lg / 4) / V
        out[f"lg_n{int(np.log10(nb))}_S"] = S
        print(f"loop-gap n_bar={nb:.0e}: best S={dB(S.max()):.2f} dB", flush=True)
    out["lg_t"] = t_eval
    # twisting at the same operating point for three line shapes (comparison, light grid)
    from cavsqueeze.protocols import optimal_squeezing
    with Timer("twisting comparison"):
        N = 1e10
        p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, 3e7, T=T_BATH, T2=T2_SPIN)
        for shape, frac in [("voigt", LORENTZ_FRACTION), ("gauss", 0.0), ("lorentz", 1.0)]:
            ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT, lorentz_fraction=frac)
            b = optimal_squeezing(p, ens, 1e-5, 2e-3, echo=True, rtol=1e-6, n_coarse=10, max_fine=24)
            out[f"twist_{shape}"] = np.array([b["xi2"], b["t"]])
            print(f"twisting {shape}: {-dB(b['xi2']):.2f} dB at {b['t']*1e6:.0f} us", flush=True)
    # measurement route at the SC operating point for the three line shapes (direct protocol)
    with Timer("line shapes"):
        N = 1e10
        p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, 3e7, T=T_BATH, T2=T2_SPIN)
        times = np.linspace(0, T_END, 4001)
        t_eval = np.geomspace(1e-5, T_END, 121)
        for shape in ["gaussian", "lorentzian"]:
            traj = meanfield_trajectory(p, N, times, shape=shape)
            for nb in [1e8, 1e9]:
                V, Gm = conditional_variance(p, N, traj, nb, 0.5, T_END, t_eval)
                W = (N / 4) / V * np.interp(t_eval, times, traj["contrast"]) ** 2
                out[f"shape_{shape}_n{int(np.log10(nb))}_W"] = W
                print(f"measurement, {shape} line, n_bar={nb:.0e}: best gain {dB(np.nanmax(W)):.2f} dB, C(1 ms)={np.interp(1e-3, times, traj['contrast']):.3f}", flush=True)
            out[f"shape_{shape}_C1ms"] = np.interp(1e-3, times, traj["contrast"])
    # cumulant check of the unconditional growth of Var(Jz) for a dephased ensemble (Delta = 1 GHz)
    with Timer("cumulant check, dephased"):
        N = 1e9
        p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, 1e9, T=T_BATH, T2=T2_SPIN)
        ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
        rt = Rates.from_params(p, ens)
        st = css_x(rt.M)
        st.vs = np.tile(X, (rt.K, 1))
        ts = np.array([2e-4, 5e-4, 1e-3, 2e-3])
        sts = evolve(st, rt, ts[-1], t_eval=ts, rtol=1e-7)
        rows = []
        for t, s_ in zip(ts, sts):
            J, Cov, S1, S2 = collective_moments(s_, rt.n, spec_n=rt.spec_n)
            rows.append([t, Cov[2, 2] - N / 4, p.Gamma_SR * N / 4 * t, 2 * abs(J[0] + 1j * J[1]) / N])
            print(f"   dephased, t={t*1e3:.1f} ms: Var(Jz)-N/4 = {Cov[2,2]-N/4:.3e} (cumulant) vs Gamma_SR N t/4 = {p.Gamma_SR*N/4*t:.3e}; contrast {rows[-1][3]:.3f}", flush=True)
        out["check_dephased_rows"] = np.array(rows)
    # cumulant check of the unconditional growth of Var(Jz) (locked case, short interval)
    with Timer("cumulant check"):
        N = 1e10
        p = from_hz(GN_HZ / np.sqrt(N), KAPPA_HZ, 3e7, T=T_BATH, T2=T2_SPIN)
        ens = standard_ensemble(N, p.chi * N, "voigt", GRID_LIGHT)
        rt = Rates.from_params(p, ens)
        st = css_x(rt.M)
        st.vs = np.tile(X, (rt.K, 1))
        ts = np.array([2e-5, 5e-5, 1e-4, 2e-4])
        sts = evolve(st, rt, ts[-1], t_eval=ts, rtol=1e-7)
        rows = []
        for t, s in zip(ts, sts):
            J, Cov, S1, S2 = collective_moments(s, rt.n, spec_n=rt.spec_n)
            D_pred = p.Gamma_SR * (p.n_th + 1) * (N / 2) ** 2 * t
            rows.append([t, Cov[2, 2] - N / 4, D_pred, J[2]])
            print(f"   t={t*1e6:.0f} us: Var(Jz)-N/4 = {Cov[2,2]-N/4:.3e} (cumulant) vs {D_pred:.3e} (Gamma_SR |J_perp|^2 t)", flush=True)
        out["check_rows"] = np.array(rows)
    save("measurement", **out)
