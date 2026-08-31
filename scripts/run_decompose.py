"""Where the noise sits at the optimum: locked core against unlocked wings.

At the optimum of each scan point the collective variance along the squeezed
quadrature is split by detuning class into the part carried by spins inside the
many-body gap (|delta| <= chi N), the part carried by spins outside it, and the
cross term.  The same split is applied to the mean spin.  This turns the wing
floor of the main text from a fit into a measured decomposition: the floor is
the projection noise of the spins the interaction never locks.
"""
from common import *  # noqa
from cavsqueeze.cumulant import (Rates, collective_moments, to_cartesian,
                                 transverse_variances)
from cavsqueeze.protocols import optimal_squeezing, twist, css_x
from concurrent.futures import ProcessPoolExecutor

N = 1e10
KAPPA = 1e4


def split_variance(st, ens, inside_mask, n_hat):
    """Variance along n_hat carried by inside-inside, cross and outside-outside
    class pairs, plus the single-spin (same-class) part of each group."""
    pops = ens.n
    v, C = to_cartesian(st)
    nh = np.asarray(n_hat, float)

    def quad(A):
        return float(nh @ A @ nh)

    M = len(pops)
    ins = np.asarray(inside_mask, bool)
    out = ~ins
    # pair (inter-spin) part, per group of class pairs
    full = np.einsum("m,n,mnab->ab", pops, pops, C)
    self_pair = np.einsum("m,mmab->ab", pops, C)

    def pair_block(mask_m, mask_n):
        pm = np.where(mask_m, pops, 0.0)
        pn = np.where(mask_n, pops, 0.0)
        return np.einsum("m,n,mnab->ab", pm, pn, C)

    P_ii = pair_block(ins, ins) - np.einsum("m,mmab->ab", np.where(ins, pops, 0.0), C)
    P_oo = pair_block(out, out) - np.einsum("m,mmab->ab", np.where(out, pops, 0.0), C)
    P_io = pair_block(ins, out) + pair_block(out, ins)
    # single-spin part, per group
    def same_block(mask):
        p = np.where(mask, pops, 0.0)
        return np.sum(p) * np.eye(3) - np.einsum("m,ma,mb->ab", p, v, v)

    S_i, S_o = same_block(ins), same_block(out)
    # spectators: free spins, always outside the gap, uncorrelated
    K = st.vs.shape[0]
    S_spec = np.zeros((3, 3))
    if K:
        sp = ens.spec_n
        S_spec = np.sum(sp) * np.eye(3) - np.einsum("k,ka,kb->ab", sp, st.vs, st.vs)
    var_ii = 0.25 * (quad(P_ii.real) + quad(S_i))
    var_oo = 0.25 * (quad(P_oo.real) + quad(S_o) + quad(S_spec))
    var_io = 0.25 * quad(P_io.real)
    # mean spin, per group
    J_i = 0.5 * (np.where(ins, pops, 0.0) @ v)
    J_o = 0.5 * (np.where(out, pops, 0.0) @ v)
    if K:
        J_o = J_o + 0.5 * (ens.spec_n @ st.vs)
    return var_ii, var_oo, var_io, J_i, J_o


def job(args):
    ratio, chiN_hz, shape = args
    Delta = ratio * KAPPA / 2
    gN = np.sqrt(chiN_hz * Delta)
    p = from_hz(gN / np.sqrt(N), KAPPA, Delta, T=0.0, T2=T2_SPIN)
    ens = standard_ensemble(N, p.chi * N, shape, GRID_SCAN)
    best = optimal_squeezing(p, ens, 2e-6, 2e-3, echo=True, rtol=1e-6, n_coarse=10, max_fine=24)
    rt = Rates.from_params(p, ens)
    st = twist(css_x(ens.M), rt, best["t"], echo=True, rtol=1e-8)
    J, Cov, S1, S2 = collective_moments(st, ens.n, spec_n=ens.spec_n)
    vmin, vmax, ang, Jn = transverse_variances(J, Cov)
    # unit vector along the squeezed quadrature
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    n_hat = np.cos(ang) * e1 + np.sin(ang) * e2
    gap = abs(p.chi * N)                      # many-body gap, rad/s
    inside = np.abs(ens.delta) <= gap
    var_ii, var_oo, var_io, J_i, J_o = split_variance(st, ens, inside, n_hat)
    n_in = float(ens.n[inside].sum())
    n_out = float(ens.n[~inside].sum() + ens.spec_n.sum())
    f_out = n_out / (n_in + n_out)
    xi2 = float(vmin * S1**2 / (Jn**2 * S2))
    contrast = float(2.0 * np.hypot(J[0], J[1]) / S1)
    # what the locked core alone would deliver: same state, wings removed from
    # both the noise and the mean spin
    pops_in = np.where(inside, ens.n, 0.0)
    J_c, Cov_c, S1_c, S2_c = collective_moments(st, pops_in)
    vmin_c, _, _, Jn_c = transverse_variances(J_c, Cov_c)
    xi2_core = float(vmin_c * S1_c**2 / (Jn_c**2 * S2_c)) if Jn_c > 0 else np.inf
    return dict(ratio=ratio, chiN_hz=chiN_hz, shape=shape, xi2=xi2, t=best["t"],
                contrast=contrast, f_out=f_out, var_tot=float(vmin),
                var_ii=var_ii, var_oo=var_oo, var_io=var_io,
                Jn=float(Jn), J_i=float(np.linalg.norm(J_i)), J_o=float(np.linalg.norm(J_o)),
                N_tot=float(n_in + n_out), xi2_core=xi2_core, n_out=n_out)


if __name__ == "__main__":
    shapes = ["voigt", "gaussian", "lorentzian"]
    ratios = [100, 1000, 10000]
    chiN = [1, 2, 4, 8, 16]
    jobs = [(r, c * GAMMA_INH_HZ, s) for s in shapes for r in ratios for c in chiN]
    rows = []
    with ProcessPoolExecutor(2) as ex, Timer("decompose"):
        for r in ex.map(job, jobs):
            print(r, flush=True)
            rows.append(r)
    keys = ["ratio", "chiN_hz", "xi2", "t", "contrast", "f_out", "var_tot",
            "var_ii", "var_oo", "var_io", "Jn", "J_i", "J_o", "N_tot",
            "xi2_core", "n_out"]
    arr = np.array([[r[k] for k in keys] for r in rows], float)
    shp = np.array([{"voigt": 0, "gaussian": 1, "lorentzian": 2}[r["shape"]] for r in rows], float)
    save("decompose", rows=arr, shape=shp, keys=np.array(keys))
