"""The gap-protected coherence as a synchronisation order parameter.

Mean-field reduction (Sec. S6).  With the resonator eliminated, each spin
precesses in the field of the collective spin.  Writing the Bloch vector of a
spin of detuning delta as (rho cos phi, rho sin phi, z) with rho = sqrt(1-z^2),
and psi = phi - Psi the phase relative to the mean spin, the mean-field
equations of the solver reduce exactly to

    d psi / dt = delta - Omega z cos(psi) / rho,     dz / dt = Omega rho sin(psi),

with the locking field  Omega = chi N R  and R the Ramsey contrast.  These have
the conserved energy  E = (delta/2) z + (Omega/2) rho cos(psi), so along every
orbit

    rho cos(psi) = 1 - (delta/Omega) z                                   (exact)

for the coherent-spin-state initial condition (psi = 0, z = 0).  The same
integral gives the locking criterion: psi can reach pi only if |delta| >= Omega,
so spins inside the gap librate and spins outside it run.  A librating orbit
runs between z = 0 and z = 2 Omega delta/(Omega^2 + delta^2), and its average is
half of that to 0.2 per cent (checked here), so averaging the exact relation
over the line closes into

    R = int p(delta) Omega^2 / (Omega^2 + delta^2) d delta,   Omega = chi N R,

a Lorentzian smoothing of the line at the width of the locking field.  For a
Lorentzian line of FWHM gamma this has the closed form R = 1 - gamma/(2 chi N),
with locking setting in at chi N = gamma/2.

This script (a) checks the orbit average that closes the equation, (b) solves
the self-consistency for the three line shapes, and (c) compares it with the
mean-field solver of the package.
"""
from common import *  # noqa
from cavsqueeze.ensemble import lineshape
from cavsqueeze.protocols import ramsey_meanfield
from concurrent.futures import ProcessPoolExecutor
from scipy import optimize
from scipy.integrate import solve_ivp

N = 1e10
KAPPA = 1e4
RATIO = 1.0e5           # 2 Delta/kappa; collective emission negligible over the window
FW = TWO_PI * GAMMA_INH_HZ


# --- (a) the orbit average that closes the self-consistency -----------------
def orbit_average(dratio, T=400.0, n=200001):
    """Time averages of z and of rho cos(psi) on the orbit through (0,0), Omega=1."""
    def rhs(t, y):
        psi, z = y
        rho = np.sqrt(max(1e-16, 1.0 - z * z))
        return [dratio - z * np.cos(psi) / rho, rho * np.sin(psi)]
    s = solve_ivp(rhs, (0, T), [0.0, 0.0], t_eval=np.linspace(0, T, n),
                  rtol=1e-11, atol=1e-13, method="DOP853")
    psi, z = s.y
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    running = bool(np.abs(psi).max() > np.pi)
    return float(z.mean()), float((rho * np.cos(psi)).mean()), running


# --- (b) the self-consistency ----------------------------------------------
def _grid(fwhm, ngrid):
    pos = np.geomspace(1e-4 * fwhm, 300 * fwhm, ngrid // 2)
    return np.concatenate([-pos[::-1], [0.0], pos])


def R_law(chiN, shape, fwhm=FW, eta=LORENTZ_FRACTION, ngrid=200001):
    """Largest root of R = int p(d) Omega^2/(Omega^2+d^2) dd with Omega = chi N R."""
    ls = lineshape(shape, fwhm, eta)
    d = _grid(fwhm, ngrid)
    p = np.asarray(ls.pdf(d), float)

    def rhs(R):
        if R <= 0:
            return 0.0
        W = chiN * R
        return float(np.trapezoid(p * W * W / (W * W + d * d), d))

    g = lambda R: rhs(R) - R
    if g(1.0) > 0:
        return 1.0
    lo = None
    for R in np.linspace(1.0, 1e-5, 800):
        if g(R) > 0:
            lo = R
            break
    return 0.0 if lo is None else float(optimize.brentq(g, lo, 1.0, xtol=1e-12))


def locking_threshold(shape, fwhm=FW, eta=LORENTZ_FRACTION):
    """Threshold from R -> 0.  The kernel Omega^2/(Omega^2+d^2) integrates to pi Omega
    and narrows onto delta = 0, so the right-hand side tends to pi chi N R p(0) and a
    non-zero root first appears at chi N_c = 1/(pi p(0)); half the Kuramoto value
    2/(pi p(0)), because a librating spin keeps part of its projection on the mean
    spin while a running Kuramoto oscillator keeps none."""
    ls = lineshape(shape, fwhm, eta)
    p0 = float(np.atleast_1d(ls.pdf(0.0))[0])
    return 1.0 / (np.pi * p0)


# --- (c) the mean-field solver ---------------------------------------------
def job(args):
    chiN_hz, shape = args
    Delta = RATIO * KAPPA / 2
    gN = np.sqrt(chiN_hz * Delta)
    p = from_hz(gN / np.sqrt(N), KAPPA, Delta, T=0.0, T2=np.inf)
    ens = standard_ensemble(N, p.chi * N, shape, GRID_STD)
    t = np.linspace(0.0, 80.0 / GAMMA_INH_HZ, 8001)
    c = ramsey_meanfield(p, ens, t)["contrast"]
    tail = c[len(c) // 2:]
    return dict(chiN_hz=chiN_hz, shape=shape, plateau=float(tail.mean()),
                spread=float(tail.std()), R_law=R_law(TWO_PI * chiN_hz, shape))


if __name__ == "__main__":
    print("(a) orbit average: <z> against z_max/2, and the energy identity")
    orb = []
    for dr in [0.2, 0.5, 0.8, 0.95, 1.05, 1.5, 3.0]:
        zbar, proj, run = orbit_average(dr)
        zmax_half = dr / (1.0 + dr * dr)
        orb.append([dr, zbar, zmax_half, proj, 1.0 - dr * zbar, 1.0 if run else 0.0])
        print(f"  delta/Omega={dr:5.2f}  <z>={zbar:.5f}  z_max/2={zmax_half:.5f}  "
              f"ratio={zbar/zmax_half:.4f}  <rho cos psi>={proj:.5f}  "
              f"1-(delta/Omega)<z>={1.0-dr*zbar:.5f}  {'running' if run else 'locked'}", flush=True)

    print("\n(b) locking threshold of each line shape (units of the FWHM)")
    thr = {}
    for sh in ["lorentzian", "voigt", "gaussian"]:
        thr[sh] = locking_threshold(sh) / FW
        print(f"  {sh:<11} chi N_c = {thr[sh]:.4f} x gamma_inh", flush=True)

    print("\n(c) self-consistency against the mean-field solver")
    ratios = [0.7, 0.9, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0, 8.0, 16.0]
    jobs = [(r * GAMMA_INH_HZ, s) for s in ["lorentzian", "voigt", "gaussian"] for r in ratios]
    rows = []
    with ProcessPoolExecutor(2) as ex, Timer("locking"):
        for r in ex.map(job, jobs):
            print("  ", r, flush=True)
            rows.append(r)
    keys = ["chiN_hz", "plateau", "spread", "R_law"]
    arr = np.array([[r[k] for k in keys] for r in rows], float)
    shp = np.array([{"voigt": 0, "gaussian": 1, "lorentzian": 2}[r["shape"]] for r in rows], float)
    save("locking", rows=arr, shape=shp, keys=np.array(keys),
         orbit=np.array(orb, float),
         threshold=np.array([thr["voigt"], thr["gaussian"], thr["lorentzian"]], float))
