"""Zero-field hyperfine levels of the 171Yb3+:CaWO4 ground state and the
matrix elements that couple them to a microwave magnetic field.

Ground-state spin Hamiltonian at zero field (Tiranov et al., arXiv:2504.01592):
    H = A_perp (Ix Sx + Iy Sy) + A_par Iz Sz,
with A_par/h = 0.787 GHz and A_perp/h = 3.08384 GHz (their values).  The
resonator field of the loop-gap device is along the crystal c axis (Fukumori et
al., arXiv:2604.26909, Sec. S3), so it couples through Sz with g_par = 1.08.

The script prints the four level energies, the transition frequencies from the
two clock levels to the other two levels, and the Sz and Sx matrix elements
between all levels, which show that a c-axis field does not connect the clock
levels to the other two levels at all.  Values are written to
data/hyperfine_levels.json.
"""
from common import *  # noqa

A_PAR = 0.787e9
A_PERP = 3.08384e9


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
    out = dict(levels_GHz=((E - E[0]) / 1e9).tolist(), f14_GHz=f14, f12_GHz=f12, f24_GHz=f24,
               Sz_abs=Szm.tolist(), Sx_abs=Sxm.tolist(), A_par_GHz=A_PAR / 1e9, A_perp_GHz=A_PERP / 1e9)
    save_json("hyperfine_levels", out)
