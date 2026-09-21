"""Zero-field hyperfine levels of the 171Yb3+:CaWO4 ground state and the
matrix elements that couple them to a microwave magnetic field.

Ground-state spin Hamiltonian at zero field (Tiranov et al., arXiv:2504.01592):
    H = A_perp (Ix Sx + Iy Sy) + A_par Iz Sz,
with A_par/h = 0.787 GHz and A_perp/h = 3.08384 GHz (their values).  The
resonator field of the loop-gap device is along the crystal c axis (Fukumori et
al., arXiv:2604.26909, Sec. S3), so it couples through Sz with g_par = 1.08.
A field component perpendicular to c, of relative size eps, couples through
g_perp Sx.  The ground-state g values are those tabulated by Tiranov et al.
(Table S1 of arXiv:2504.01592): g_perp = 3.916, g_par = 1.053.

The script prints the four level energies, the transition frequencies from the
two clock levels to the other two levels, and the Sz and Sx matrix elements
between all levels, which show that a c-axis field does not connect the clock
levels to the other two levels at all.  Values are written to
data/hyperfine_levels.json.
"""
from common import *  # noqa

A_PAR = 0.787e9
A_PERP = 3.08384e9
G_PAR_T, G_PERP_T = 1.053, 3.916   # Tiranov et al., Table S1, ground state
DELTA_LG = 22e6                    # loop-gap spin-resonator detuning (Hz)


def spin_half():
    sx = 0.5 * np.array([[0, 1], [1, 0]], complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], complex)
    return sx, sy, sz


if __name__ == "__main__":
    sx, sy, sz = spin_half()
    I2 = np.eye(2)
    Sx, Sy, Sz = (np.kron(o, I2) for o in (sx, sy, sz))
    Ix, Iy, Iz = (np.kron(I2, o) for o in (sx, sy, sz))
    H = A_PERP * (Ix @ Sx + Iy @ Sy) + A_PAR * (Iz @ Sz)
    E, V = np.linalg.eigh(H)
    order = np.argsort(E)
    E, V = E[order], V[:, order]
    names = ["1", "2", "3", "4"]
    print("levels (GHz relative to lowest):", np.round((E - E[0]) / 1e9, 5))
    Szm = np.abs(V.conj().T @ Sz @ V)
    Sxm = np.abs(V.conj().T @ Sx @ V)
    print("|<i|Sz|j>|:\n", np.round(Szm, 4))
    print("|<i|Sx|j>|:\n", np.round(Sxm, 4))
    f14 = (E[3] - E[0]) / 1e9
    f12 = (E[1] - E[0]) / 1e9
    f24 = (E[3] - E[1]) / 1e9
    print(f"clock transition 1-4: {f14:.5f} GHz; 1-2,3: {f12:.4f} GHz; 2,3-4: {f24:.4f} GHz")
    # misaligned field: coupling of |1>,|4> to |2>,|3> relative to the clock
    # coupling, and the dispersive shift it causes relative to chi
    sx12 = Sxm[0, 1]
    sz14 = Szm[0, 3]
    eps_coef = (G_PERP_T * sx12) / (G_PAR_T * sz14)      # coupling ratio per unit eps
    f_res = f14 * 1e9 + DELTA_LG                           # resonator frequency
    d1 = f_res - f12 * 1e9                                 # resonator detuning from 1-2,3
    d4 = f_res - f24 * 1e9                                 # resonator detuning from 2,3-4
    per_level = eps_coef**2 * DELTA_LG / d1
    total = 2 * eps_coef**2 * DELTA_LG / d1 + 2 * eps_coef**2 * DELTA_LG / d4
    eps15 = np.tan(np.radians(15.0))
    print(f"g_perp/g_par = {G_PERP_T / G_PAR_T:.3f}; coupling ratio = {eps_coef:.3f} eps")
    print(f"detunings from the other levels: {d1 / 1e9:.3f} and {d4 / 1e9:.3f} GHz")
    print(f"shift relative to chi: {per_level:.3f} eps^2 per level, {total:.3f} eps^2 in total; "
          f"{100 * total * eps15**2:.1f}% at 15 degrees")
    out = dict(levels_GHz=((E - E[0]) / 1e9).tolist(), f14_GHz=f14, f12_GHz=f12, f24_GHz=f24,
               Sz_abs=Szm.tolist(), Sx_abs=Sxm.tolist(), A_par_GHz=A_PAR / 1e9, A_perp_GHz=A_PERP / 1e9,
               g_par_Tiranov=G_PAR_T, g_perp_Tiranov=G_PERP_T, coupling_ratio_per_eps=eps_coef,
               detuning_1_23_GHz=d1 / 1e9, detuning_23_4_GHz=d4 / 1e9,
               shift_per_level_eps2=per_level, shift_total_eps2=total,
               shift_total_15deg=total * eps15**2)
    save_json("hyperfine_levels", out)
