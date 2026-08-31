"""Extract every number quoted in the manuscript from data/*.npz -> data/numbers.json
and fill the [[PLACEHOLDER]] tokens of paper/main.tex (writes paper/main_filled.tex)."""
from common import *  # noqa
import re

num = {}


def first_dB(x):
    return float(dB(x))


# ---------------- benchmark ----------------
b = load("benchmark")
if b is not None:
    t = b["t"]
    for c in [4000, 7000]:
        con = b[f"mf_voigt_{c}"]
        idx = np.where(con < np.exp(-1))[0]
        num[f"T1E_{c//1000}"] = round(float(t[idx[0]] * 1e3), 2) if len(idx) else None
        num[f"T1E_{c//1000}B"] = round(float(t[idx[0]] * 1e3), 1) if len(idx) else None
    tc = b["cum_t"]
    m = tc <= 3.0e-3
    num["ANTISQ_7"] = round(first_dB(np.interp(3e-3, tc, b["cum_varmax_7000"])), 1)
    num["SQ_7"] = round(-first_dB(np.min(b["cum_varmin_7000"][m])), 1)
    num["SQ_7_T"] = round(float(tc[np.argmin(b["cum_varmin_7000"][m])] * 1e3), 2)

# ---------------- loop-gap ----------------
d = load("loopgap")
if d is not None and "a_xi_homogeneous_echo" in d:
    t = d["t_list"]
    for shape, key in [("homogeneous", "HOMO"), ("gaussian", "GAUSS"), ("voigt", "VOIGT"), ("lorentzian", "LOR")]:
        for e in ["echo", "noecho"]:
            k = f"a_xi_{shape}_{e}"
            if k in d:
                xi = d[k]
                num[f"LG_{key}_{e.upper()}"] = round(first_dB(np.nanmin(xi)), 1)
                num[f"LG_{key}_{e.upper()}_T"] = round(float(t[np.nanargmin(xi)] * 1e6), 0)
        best_e = min(num.get(f"LG_{key}_ECHO", 0), num.get(f"LG_{key}_NOECHO", 0))
        num[f"LG_{key}"] = round(best_e, 1)
        num[f"LG_{key}_ABS"] = round(-best_e, 1)
    num["LG_T"] = num.get("LG_VOIGT_NOECHO_T") if num.get("LG_VOIGT_NOECHO", 0) < num.get("LG_VOIGT_ECHO", 0) else num.get("LG_VOIGT_ECHO_T")
    num["LG_HOMO_T"] = num.get("LG_HOMO_ECHO_T")
    num["LG_GAUSS_LOSS"] = round(num["LG_GAUSS"] - num["LG_HOMO"], 1)
    num["LG_GAUSS_LOSS_ABS"] = round(abs(num["LG_GAUSS"] - num["LG_HOMO"]), 1)
    num["LG_LOR_LOSS_ABS"] = round(abs(num["LG_LOR"] - num["LG_HOMO"]), 1)
    num["LG_LOR_LOSS"] = round(num["LG_LOR"] - num["LG_HOMO"], 1)
    num["LG_VOIGT_LOSS"] = round(num["LG_VOIGT"] - num["LG_HOMO"], 1)
    chiN0 = 6134.98  # Hz, chi N0/2pi at N0 = 6e14
    num["LOR_TAIL"] = round(100 * (1 - 2 / np.pi * np.arctan(2 * chiN0 / GAMMA_INH_HZ)), 0)
    if "b_rows" in d:
        rows = d["b_rows"]
        for N, tag in [(6e14, ""), (1.35e15, "2")]:
            r = rows[(rows[:, 0] == N) & (rows[:, 2] == 1)]
            k = np.argmin(r[:, 3])
            num[f"LG_DOPT{tag}"] = round(float(r[k, 1] / 1e6), 0)
            num[f"LG_BEST{tag}"] = round(first_dB(r[k, 3]), 1)
            num[f"LG_BEST{tag}_ABS"] = round(-first_dB(r[k, 3]), 1)
            num[f"LG_BEST{tag}_T"] = round(float(r[k, 4] * 1e6), 0)
            r22 = r[np.isclose(r[:, 1], 22e6)]
            if len(r22):
                num[f"LG_AT22{tag}"] = round(first_dB(np.min(r22[:, 3])), 1)
            num[f"LG_DELTA_TABLE{tag}"] = sorted([[float(a / 1e6), round(first_dB(x), 1)] for a, x in zip(r[:, 1], r[:, 3])])
        num["LG_GAIN"] = round(num["LG_AT22"] - num["LG_BEST"], 1)
        num["LG_AT22_ABS"] = abs(num["LG_AT22"])
    if "LG_LOR_NOECHO" in num:
        num["LG_LOR_NOECHO_ABS"] = abs(num["LG_LOR_NOECHO"])
        num["LG_GAIN2"] = round(num["LG_AT222"] - num["LG_BEST2"], 1)
    if "c_rows" in d:
        rows = d["c_rows"]

        def best_per_N(r):
            out = []
            for Nv in np.unique(r[:, 0]):
                rr = r[r[:, 0] == Nv]
                out.append(rr[np.argmin(rr[:, 3])])
            return np.array(out)

        rv = best_per_N(rows[rows[:, 2] == 1])
        rh = best_per_N(rows[rows[:, 2] == 0])
        diff = dB(rv[:, 3]) - dB(rh[:, 3])
        within = np.where(diff < 1.0)[0]
        num["LG_NRATIO"] = round(float(rv[within[0], 6] / GAMMA_INH_HZ), 0) if len(within) else None
        num["LG_N_TABLE"] = [[float(a), round(first_dB(x), 1), round(first_dB(y), 1)] for a, x, y in zip(rv[:, 0], rv[:, 3], rh[:, 3])]

# ---------------- the collective-emission law and its perturbative counterpart ----------------
_v = load("validation")
if _v is not None:
    _A = float(_v["c_prefactor_xi"])
    num["PREFACTOR"] = round(_A, 2)
    # Lewis-Swan et al.: 3/2^{2/3} (Gamma/chi)^{2/3} = 3.00 (kappa/2Delta)^{2/3}
    _pert = 3.0 * 2.0 ** (-2.0 / 3.0) * 2.0 ** (2.0 / 3.0)
    num["PREFACTOR_PERT"] = round(_pert, 2)
    num["PERT_GAIN"] = round(float(10 * np.log10(_pert / _A)), 1)
    num["LAW1_FIT_ERR"] = round(float(100 * np.max(np.abs(_v["c_xi_opt"] / (_A * (1 / _v["c_ratio"]) ** (2 / 3)) - 1))), 0)

# ---------------- synchronisation: the locking law and its threshold ----------------
from cavsqueeze.ensemble import lineshape as _lineshape
from scipy import optimize as _opt

_SHNAME = {0: "voigt", 1: "gaussian", 2: "lorentzian"}
_FW = TWO_PI * GAMMA_INH_HZ


def _dgrid(fwhm, n=200001):
    pos = np.geomspace(1e-4 * fwhm, 300 * fwhm, n // 2)
    return np.concatenate([-pos[::-1], [0.0], pos])


def _R_law(chiN, shape, fwhm=_FW, eta=LORENTZ_FRACTION):
    """Largest root of R = int p(d) Omega^2/(Omega^2+d^2) dd, Omega = chi N R."""
    ls = _lineshape(shape, fwhm, eta)
    dd = _dgrid(fwhm)
    pd = np.asarray(ls.pdf(dd), float)

    def rhs(R):
        if R <= 0:
            return 0.0
        W = chiN * R
        return float(np.trapezoid(pd * W * W / (W * W + dd * dd), dd))

    f = lambda R: rhs(R) - R
    if f(1.0) > 0:
        return 1.0
    lo = None
    for R in np.linspace(1.0, 1e-5, 800):
        if f(R) > 0:
            lo = R
            break
    return 0.0 if lo is None else float(_opt.brentq(f, lo, 1.0, xtol=1e-12))


def _tail(x_rad, shape, fwhm=_FW, eta=LORENTZ_FRACTION):
    """Mass of the line outside |delta| > x."""
    return float(2.0 * (1.0 - _lineshape(shape, fwhm, eta).cdf(x_rad)))


def _threshold(shape, fwhm=_FW, eta=LORENTZ_FRACTION):
    """chi N_c = 1/(pi p(0)); half the Kuramoto value 2/(pi p(0))."""
    p0 = float(np.atleast_1d(_lineshape(shape, fwhm, eta).pdf(0.0))[0])
    return 1.0 / (np.pi * p0)


for _sh, _tag in [("voigt", "V"), ("gaussian", "G"), ("lorentzian", "L")]:
    num[f"CHINC_{_tag}"] = round(_threshold(_sh) / _FW, 3)
num["CHINC_LG_RATIO"] = round(float(TWO_PI * 6.1e3 / _threshold("voigt")), 1)

_lk = load("locking")
if _lk is not None:
    _rows, _shp = _lk["rows"], _lk["shape"]
    _k = {k: i for i, k in enumerate(list(_lk["keys"]))}
    _dev = np.abs(_rows[:, _k["R_law"]] - _rows[:, _k["plateau"]])
    num["SYNC_MEAN_DEV"] = float(f"{np.mean(_dev):.4f}")
    num["SYNC_MAX_DEV"] = float(f"{np.max(_dev):.4f}")
    num["SYNC_NPTS"] = int(len(_dev))
    _orb = _lk["orbit"]                       # dratio, <z>, zmax/2, proj, 1-dr<z>, running
    num["ORBIT_MAX_ERR"] = round(float(100 * np.max(np.abs(_orb[:, 1] / _orb[:, 2] - 1))), 1)
    num["ORBIT_ID_ERR"] = float(f"{np.max(np.abs(_orb[:, 3] - _orb[:, 4])):.1e}")
    # table: contrast from the law against the mean-field plateau, shapes side by side
    _xs = np.unique(np.round(_rows[:, _k["chiN_hz"]] / GAMMA_INH_HZ, 3))
    _lines = []
    for _x in _xs:
        _cells = []
        for _code in (2, 0, 1):                    # Lorentzian, Voigt, Gaussian
            _m = (_shp == _code) & (np.abs(_rows[:, _k["chiN_hz"]] / GAMMA_INH_HZ - _x) < 1e-6)
            if not _m.any():
                _cells += ["--", "--"]
                continue
            _j = int(np.flatnonzero(_m)[0])
            _cells += ["%.3f" % _rows[_j, _k["plateau"]], "%.3f" % _rows[_j, _k["R_law"]]]
        _lines.append("%.1f & " % _x + " & ".join(_cells) + r" \\")
    num["SYNC_TABLE"] = "\n".join(_lines)

# ---------------- the two limits, tested on the decomposition scan ----------------
_dc = load("decompose")
if _dc is not None:
    _rows, _shp = _dc["rows"], _dc["shape"]
    _k = {k: i for i, k in enumerate(list(_dc["keys"]))}
    _cand = {a: [] for a in "ACE"}
    _gauss, _cost, _cost_g = [], [], []
    for _r, _s in zip(_rows, _shp):
        _sh = _SHNAME[int(_s)]
        _ratio, _chiN = _r[_k["ratio"]], TWO_PI * _r[_k["chiN_hz"]]
        _xi2, _core = _r[_k["xi2"]], _r[_k["xi2_core"]]
        _R = _R_law(_chiN, _sh)
        _res = 1.43 * (1.0 / _ratio) ** (2.0 / 3.0)
        _cand["A"].append(dB(max(_res, _tail(_chiN, _sh))) - dB(_xi2))
        _cand["C"].append(dB(max(_res, _tail(_chiN * max(_R, 1e-12), _sh))) - dB(_xi2))
        _cand["E"].append(dB(max(_res, 1.0 - _R)) - dB(_xi2))
        if int(_s) == 1:
            _gauss.append(abs(float(_cand["C"][-1])))
            _cost_g.append(float(dB(_xi2) - dB(_core)))
        _cost.append(float(dB(_xi2) - dB(_core)))
    num["LAW2_MEAN_DEV"] = round(float(np.mean(np.abs(_cand["C"]))), 2)
    num["LAW2_MAX_DEV"] = round(float(np.max(np.abs(_cand["C"]))), 2)
    num["LAW2_NPTS"] = int(len(_cand["C"]))
    num["LAW2_GAUSS_DEV"] = round(float(np.max(_gauss)), 2)
    num["LAW2_NAIVE_DEV"] = round(float(np.mean(np.abs(_cand["A"]))), 2)
    num["LAW2_ALT_DEV"] = round(float(np.mean(np.abs(_cand["E"]))), 2)
    num["WINGS_COST_MAX"] = round(float(np.max(_cost)), 1)
    num["WINGS_COST_GAUSS"] = round(float(np.max(np.abs(_cost_g))), 2)

# ---------------- scaling ----------------
s = load("scaling")
if s is not None:
    rows = s["a_rows"]
    sat = []
    def best_over_echo(r):
        out = []
        for c in np.unique(r[:, 1]):
            rr = r[r[:, 1] == c]
            out.append(rr[np.argmin(rr[:, 3])])
        return np.array(out)

    for ratio in [100, 1000, 10000]:
        r = best_over_echo(rows[(rows[:, 0] == ratio) & (rows[:, 2] == 0)])
        r = r[np.argsort(r[:, 1])]
        homo = 1.43 * (1 / ratio) ** (2 / 3)
        ok = np.where(dB(r[:, 3]) - dB(homo) < 1.0)[0]
        sat.append(r[ok[0], 1] / GAMMA_INH_HZ if len(ok) else np.nan)
        num[f"SCAL_{ratio}"] = [[float(a / GAMMA_INH_HZ), round(first_dB(x), 1)] for a, x in zip(r[:, 1], r[:, 3])]
    num["SAT_RATIO"] = int(np.nanmax(sat)) if np.isfinite(np.nanmax(sat)) else None
    # the two-limit law is evaluated on the decomposition scan below, which carries
    # the contrast needed for the locking field; the numbers appear as LAW2_*.
    if "b_rows" in s:
        rb = s["b_rows"]
        for ratio, tag in [(66.7, "LG"), (6000, "SC")]:
            r = rb[np.isclose(rb[:, 0], ratio)]
            r = r[np.argsort(r[:, 1])]
            x0 = dB(r[np.isclose(r[:, 1], 0.0), 3][0])
            x80 = dB(r[np.isclose(r[:, 1], 0.08), 3][0])
            x300 = dB(r[np.isclose(r[:, 1], 0.3), 3][0])
            num[f"TH_{tag}"] = round(float(x80 - x0), 1)
            num[f"TH_{tag}300"] = round(float(x300 - x0), 1)
    if "c_rows" in s:
        rc = s["c_rows"]
        r = rc[np.isclose(rc[:, 0], 6000)]
        r = r[np.argsort(r[:, 1])]
        ref = dB(r[-1, 2])
        ok = np.where(dB(r[:, 2]) - ref < 0.5)[0]
        num["T2_LIMIT"] = round(float(r[ok[0], 1] * 1e3), 0) if len(ok) else None
        num["T2_TABLE"] = [[float(a * 1e3), round(first_dB(x), 2)] for a, x in zip(r[:, 1], r[:, 2])]

# ---------------- design map ----------------
m = load("designmap")
if m is not None:
    best = m["best"]
    k = np.argmin(best[:, 4])
    num["SC_MAX"] = round(first_dB(best[k, 4]), 1)
    num["SC_MAX_ABS"] = round(-first_dB(best[k, 4]), 1)
    num["SC_MAX_T"] = round(float(best[k, 5] * 1e6), 0)
    num["SC_MAX_KAPPA"] = float(best[k, 0])
    num["SC_MAX_GN"] = float(best[k, 1])
    r = best[np.isclose(best[:, 0], 1e4) & np.isclose(best[:, 1], 1e6)]
    if len(r):
        num["SC_BEST"] = round(first_dB(r[0, 4]), 1)
        num["SC_BEST_ABS"] = round(-first_dB(r[0, 4]), 1)
        num["SC_T"] = round(float(r[0, 5] * 1e6), 0)
        num["SC_DELTA"] = round(float(r[0, 3] / 1e6), 1)
        num["SC_NU"] = float(r[0, 2])
    num["SC_TMIN"] = round(float(np.min(best[:, 5]) * 1e6), 0)
    num["SC_TMAX"] = round(float(np.max(best[:, 5]) * 1e3), 1)
    num["SC_DMIN"] = round(float(np.min(best[:, 3]) / 1e6), 1)
    num["SC_DMAX"] = round(float(np.max(best[:, 3]) / 1e6), 1)
    vals, cnt = np.unique(best[:, 2], return_counts=True)
    num["NU_OPT"] = int(vals[np.argmax(cnt)])
    num["NU_HIST"] = {int(v): int(c) for v, c in zip(vals, cnt)}

# ---------------- inhomogeneity ----------------
i = load("inhomog")
if i is not None:
    for D in [1, 2, 4, 10, 30, 100]:
        if f"xi_w_{D}" in i:
            num[f"INH_U_{D}"] = round(-first_dB(np.nanmin(i[f"xi_u_{D}"])), 1)
            num[f"INH_W_{D}"] = round(-first_dB(np.nanmin(i[f"xi_w_{D}"])), 1)
    num["INH_TABLE"] = {int(D): [round(first_dB(np.nanmin(i[f"xi_w_{int(D)}"])), 1), round(first_dB(np.nanmin(i[f"xi_u_{int(D)}"])), 1)] for D in i["D"] if f"xi_w_{int(D)}" in i}

# ---------------- readout ----------------
r = load("readout")
if r is not None:
    eps = r["eps"]
    for lab, key in [("loopgap", "LG"), ("sc_9", "SC9"), ("sc_10", "SC"), ("sc_11", "SC11")]:
        if f"{lab}_gain_tu" not in r:
            continue
        gtu = np.nanmax(r[f"{lab}_gain_tu"], axis=0)
        gpl = np.nanmax(r[f"{lab}_gain_plain"], axis=0)
        num[f"EPS_{key}"] = float(np.max(eps[dB(gtu) >= 3])) if np.any(dB(gtu) >= 3) else "unobservable"
        num[f"EPSPL_{key}"] = float(np.max(eps[dB(gpl) >= 3])) if np.any(dB(gpl) >= 3) else "unobservable"
        num[f"GAIN0_TU_{key}"] = round(first_dB(gtu[0]), 1)
        num[f"GAIN0_PL_{key}"] = round(first_dB(gpl[0]), 1)
    if "GAIN0_TU_SC" in num:
        num["TU_LOSS"] = round(num["GAIN0_PL_SC"] - num["GAIN0_TU_SC"], 1)
        if isinstance(num.get("EPS_SC"), float) and isinstance(num.get("EPSPL_SC"), float):
            num["TU_FACTOR"] = round(num["EPS_SC"] / num["EPSPL_SC"], 0)

# ---------------- corrections beyond the model ----------------
def _fmt_loss(x):
    return "below 0.01" if x < 0.01 else round(float(x), 2)


e = load("elimination")
if e is not None:
    rows = e["A_rows"]
    loss = {float(r[0]): float(dB(r[1]) - dB(r[2])) for r in rows}
    num["ELIM_LOSS_R02"] = round(loss[0.2], 2)
    num["ELIM_LOSS_R01"] = round(loss[0.1], 2)
    num["ELIM_LOSS_R005"] = round(loss[0.05], 3)
    if "A20_xi2_full" in e:
        l20 = float(dB(e["A20_xi2_full"].min()) - dB(e["A20_xi2_elim"].min()))
        num["ELIM_LOSS_R02_N20"] = round(l20, 2)
        num["ELIM_N_DIFF"] = round(abs(l20 - loss[0.2]) + 0.005, 1)
    # (g sqrt N / Delta)^2 extrapolation from the smallest ratio computed
    base = loss[0.05] / 0.05**2
    num["ELIM_LOSS_LG"] = _fmt_loss(base * 0.0167**2)
    num["ELIM_LOSS_SC30"] = _fmt_loss(base * (1 / 30) ** 2)
    num["ELIM_LOSS_SC125"] = _fmt_loss(base * 0.08**2)
    num["ELIM_TABLE"] = " \\\\\n".join(
        f"{r[0]:g} & {-dB(r[1]):.2f} & {-dB(r[2]):.2f} & {dB(r[1]) - dB(r[2]):.2f} & {100 * r[6]:.1f}\\%" for r in rows) + " \\\\"
    B = e["B_rows"]
    num["RWA_DEV_SC"] = round(100 * float(B[0, 5]), 1)
    num["RWA_DEV_LG"] = round(100 * float(B[1, 5]), 1)

v = load("reversal")
if v is not None:
    A = v["A_rows"]
    num["REV_GAIN_FREE"] = round(first_dB(A[0, 1]), 2)
    num["REV_GAIN_EMUL"] = round(first_dB(A[0, 2]), 2)
    num["REV_GAIN_JUMP"] = round(first_dB(A[0, 3]), 2)
    num["REV_EMUL_ERR"] = round(abs(first_dB(A[0, 2]) - first_dB(A[0, 3])) + 0.005, 2)
    num["REV_JUMP_LOSS"] = round(first_dB(A[0, 1]) - first_dB(A[0, 3]), 2)
    ramps = [f"{first_dB(r[3]):.2f}" for r in A[1:]]
    num["REV_GAIN_RAMPS"] = ", ".join(ramps[:-1]) + " and " + ramps[-1]
    for kap in [3000, 10000, 30000, 100000]:
        if f"B_sc_k{kap}_gain" in v and f"B_sc_k{kap}_ideal_gain" in v:
            g = np.nanmax(v[f"B_sc_k{kap}_gain"], axis=0)
            gi = np.nanmax(v[f"B_sc_k{kap}_ideal_gain"], axis=0)
            tag = {3000: "K3", 10000: "K10", 30000: "K30", 100000: "K100"}[kap]
            num[f"REV_SC_{tag}"] = round(first_dB(g[0]), 1)
            num[f"REV_SC_{tag}_IDEAL"] = round(first_dB(gi[0]), 1)
            num[f"REV_LOSS_{tag}"] = round(first_dB(gi[0]) - first_dB(g[0]), 1)
            num[f"REV_SC_LOSS_{tag}"] = num[f"REV_LOSS_{tag}"]
            eps = v[f"B_sc_k{kap}_eps"]
            num[f"REV_EPS_SC_{tag}"] = float(np.max(eps[dB(g) >= 3])) if np.any(dB(g) >= 3) else "unobservable"
    if "B_lg_gain" in v and "B_lg_ideal_gain" in v:
        g = np.nanmax(v["B_lg_gain"], axis=0)
        gi = np.nanmax(v["B_lg_ideal_gain"], axis=0)
        num["REV_LG_LOSS"] = _fmt_loss(first_dB(gi[0]) - first_dB(g[0]))

rb = load("robustness")
if rb is not None and "a_rows" in rb:
    ra = rb["a_rows"]
    fid = dict(zip(np.round(rb["a_frac"], 3), rb["a_fid_1e_us"]))
    best = {}
    ls_rows = []
    for f in np.unique(ra[:, 0]):
        re_ = ra[(ra[:, 0] == f) & (ra[:, 1] == 1)]
        rf = ra[(ra[:, 0] == f) & (ra[:, 1] == 0)]
        if len(re_) and len(rf):
            best[float(f)] = min(-first_dB(re_[0, 2]), -first_dB(rf[0, 2])) if False else max(-first_dB(re_[0, 2]), -first_dB(rf[0, 2]))
            ls_rows.append(f"{f:g} & {-first_dB(re_[0, 2]):.1f} & {re_[0, 3] * 1e6:.0f} & {-first_dB(rf[0, 2]):.1f} & {rf[0, 3] * 1e6:.0f} & {fid[round(float(f), 3)]:.0f}")
    num["LS_TABLE"] = " \\\\\n".join(ls_rows) + " \\\\"
    if 0.0 in best:
        num["LS_GAUSS"] = round(best[0.0], 1)
    if 1.0 in best:
        num["LS_LOR"] = round(best[1.0], 1)
    rbb = rb["b_rows"]
    t2_rows = []
    for T2 in np.unique(rbb[:, 0]):
        re_ = rbb[(rbb[:, 0] == T2) & (rbb[:, 1] == 1)]
        rf = rbb[(rbb[:, 0] == T2) & (rbb[:, 1] == 0)]
        if len(re_) and len(rf):
            t2_rows.append(f"{T2 * 1e3:g} & {-first_dB(re_[0, 2]):.1f} & {-first_dB(rf[0, 2]):.1f}")
            bestv = max(-first_dB(re_[0, 2]), -first_dB(rf[0, 2]))
            tag = {1e-4: "01MS", 1e-3: "1MS", 1e-2: "10MS", 0.15: "150MS"}.get(float(T2))
            if tag:
                num[f"T2_LG_{tag}"] = round(bestv, 1)
    num["T2_LG_TABLE"] = " \\\\\n".join(t2_rows) + " \\\\"
    if "T2_LG_150MS" in num and "T2_LG_1MS" in num:
        num["T2_LG_COST_1MS"] = round(num["T2_LG_150MS"] - num["T2_LG_1MS"], 1)
    rc = rb["c_rows"]
    kinds = ["ideal", "noecho", "duration", "duration_noecho", "angle", "spread"]

    def get(dev, kind, value=None):
        sub = rc[(rc[:, 0] == dev) & (rc[:, 1] == kinds.index(kind))]
        if value is not None:
            sub = sub[np.isclose(sub[:, 2], value)]
        return -first_dB(sub[0, 3]) if len(sub) else None

    lg_ideal, sc_ideal = get(0, "ideal"), get(1, "ideal")
    if lg_ideal is not None and sc_ideal is not None:
        num["SC_ECHO"] = round(sc_ideal, 1)
        num["SC_NOECHO"] = round(get(1, "noecho"), 1)
        for us, tag in [(3e-6, "3US"), (1e-5, "10US")]:
            val = get(0, "duration", us)
            if val is not None:
                num[f"PULSE_LG_{tag}"] = round(lg_ideal - val, 1)
        for us, tag in [(1e-6, "1US"), (3e-7, "03US")]:
            val = get(1, "duration", us)
            if val is not None:
                num[f"PULSE_SC_{tag}"] = round(sc_ideal - val, 1)
                num[f"PULSE_SC_{tag}_ABS"] = round(val, 1)
            valf = get(1, "duration_noecho", us)
            if valf is not None:
                num[f"PULSE_SC_FREE{tag[:-2]}_ABS"] = round(valf, 1)
        for dev, tag, ideal in [(0, "LG", lg_ideal), (1, "SC", sc_ideal)]:
            vals = [get(dev, "angle", x) for x in [0.01, 0.03, 0.1]]
            if all(x is not None for x in vals):
                num[f"ANGLE_{tag}"] = ", ".join(f"{ideal - x:.1f}" for x in vals[:-1]) + f" and {ideal - vals[-1]:.1f}"
            vals = [get(dev, "spread", x) for x in [0.02, 0.05]]
            if all(x is not None for x in vals):
                num[f"SPREAD_{tag}"] = f"{ideal - vals[0]:.1f} and {ideal - vals[1]:.1f}"
        rows_t = [("echo twist, ideal pulses", get(0, "ideal"), get(1, "ideal")),
                  ("free twisting (no $\\pi$ pulse)", get(0, "noecho"), get(1, "noecho"))]
        for (ul, us) in [(1e-6, 1e-7), (3e-6, 3e-7), (1e-5, 1e-6)]:
            rows_t.append((f"echo, $\\tau_{{\\rm p}}$ = {ul * 1e6:g} $\\mu$s (loop-gap), {us * 1e6:g} $\\mu$s (SC)", get(0, "duration", ul), get(1, "duration", us)))
            rows_t.append((f"free, $\\tau_{{\\rm p}}$ = {ul * 1e6:g} $\\mu$s (loop-gap), {us * 1e6:g} $\\mu$s (SC)", get(0, "duration_noecho", ul), get(1, "duration_noecho", us)))
        for x in [0.01, 0.03, 0.1]:
            rows_t.append((f"echo, $\\pi$ pulse angle error {100 * x:g}\\%", get(0, "angle", x), get(1, "angle", x)))
        for x in [0.02, 0.05]:
            rows_t.append((f"echo, drive-field spread {100 * x:g}\\% (unweighted spin)", get(0, "spread", x), get(1, "spread", x)))
        num["PULSE_TABLE"] = " \\\\\n".join(f"{a} & {b:.1f} & {c:.1f}" for a, b, c in rows_t if b is not None and c is not None) + " \\\\"

pc_path = os.path.join(DATA, "pulse_check.json")
if os.path.exists(pc_path):
    pc = json.load(open(pc_path))
    num["PC_EXACT"] = ", ".join(f"{100 * x:.1f}\\%" for x in pc["exact_rel_increase"][:-1]) + f" and {100 * pc['exact_rel_increase'][-1]:.1f}\\%"
    num["PC_CUM60"] = ", ".join(f"{100 * x:.1f}\\%" for x in pc["cumulant_N60_rel_increase"][:-1]) + f" and {100 * pc['cumulant_N60_rel_increase'][-1]:.1f}\\%"
    num["SPLIT_CONV"] = round(100 * abs(pc["splitting_rel_change_20_to_80"]), 2)

def fmt_sci(v):
    v = float(v)
    if v == 0:
        return "0"
    if 1e-2 <= abs(v) < 1e3:
        return f"{v:.3g}"
    e = int(np.floor(np.log10(abs(v))))
    return r"$%.1f\times10^{%d}$" % (v / 10**e, e)


dm = load("measurement")
if dm is not None:
    from cavsqueeze.resonator import HBAR
    kap = TWO_PI * 1e4
    eta = 0.5

    def mkey(N, D, name):
        return f"N{int(np.log10(N))}_D{int(D / 1e6)}_{name}"

    def best_direct(N, D, nb, et=0.5):
        W = dm[mkey(N, D, f"Wdirect_eta{et}_n{int(np.log10(nb))}")]
        return np.nanmax(W) if np.any(np.isfinite(W)) else np.nan

    def best_echo(N, D, nb, et=0.5):
        W = dm[mkey(N, D, f"Wecho_eta{et}_n{int(np.log10(nb))}")]
        return np.nanmax(W) if np.any(np.isfinite(W)) else np.nan

    # steady-state check over all locked points (Delta <= 100 MHz)
    devs = []
    for N in dm["Ns"]:
        g = TWO_PI * 1e6 / np.sqrt(N)
        for D in dm["Deltas"]:
            if D > 1.0e8:
                continue
            for et in dm["etas"]:
                for nb in dm["N_bars"]:
                    k = mkey(N, D, f"S_eta{et}_n{int(np.log10(nb))}")
                    if k in dm and nb <= 0.1 * dm[mkey(N, D, "n_crit")]:
                        S, t = dm[k], dm[mkey(N, D, "t_eval")]
                        Gm, D0 = dm[mkey(N, D, f"Gm_eta{et}_n{int(np.log10(nb))}")], dm[mkey(N, D, "D")][0]
                        tss = 3 / np.sqrt(Gm * D0)  # steady state reached, before the superradiant decay
                        if tss > 0.1 / dm[mkey(N, D, "GN")] or tss > t[-1]:
                            continue
                        Ct = np.interp(tss, dm[mkey(N, D, "t")], dm[mkey(N, D, "contrast")])
                        devs.append(np.interp(tss, t, S) * Ct / (4 * g * np.sqrt(et * nb) / kap) - 1)
    num["MS_SS_ERR"] = round(100 * float(np.max(np.abs(devs))), 1)
    num["MS_SS_NPTS"] = len(devs)
    N, D = 1e10, 3e7
    p = from_hz(1e6 / np.sqrt(N), 1e4, D, T=0.02, T2=T2_SPIN)
    g = p.g
    chi_s = g**2 / p.Delta
    for nb, tag in [(1e8, "8"), (1e9, "9")]:
        Sss = 4 * g * np.sqrt(eta * nb) / kap
        Gm = 64 * eta * chi_s**2 * nb / kap
        Dn = p.Gamma_SR * N**2 / 4
        num[f"MS_SC_N{tag}"] = round(dB(Sss), 1)
        num[f"MS_SC_T{tag}"] = round(3e3 / np.sqrt(Gm * Dn), 2)  # 3/sqrt(Gamma_m D): within 0.5% of the steady state
        num[f"MS_SC_W{tag}"] = round(dB(best_direct(N, D, nb)), 1)
        num[f"MS_PIN_{tag}"] = "%.1f pW ($%.0f$ dBm)" % (nb * dm[mkey(N, D, "P_in_per_photon")] * 1e12,
                                                        10 * np.log10(nb * dm[mkey(N, D, "P_in_per_photon")] * 1e3))
    num["MS_N9_W9"] = round(dB(best_direct(1e9, 3e7, 1e9)), 1)
    num["MS_N11_W9"] = round(dB(best_direct(1e11, 3e7, 1e9)), 1)
    num["MS_ECHO_300_FIRST"] = round(float(dm[mkey(1e9, 3e8, "C_echo")][0]), 2)
    num["MS_ECHO_300"] = round(float(np.max(dm[mkey(1e9, 3e8, "C_echo")][1:])), 2)
    ce = dm[mkey(1e9, 1e9, "C_echo")]
    num["MS_ECHO_1000"] = round(float(ce[1]), 2)
    num["MS_ECHO_1000_50"] = round(float(ce[-1]), 2)
    num["MS_N9_ECHO_1000"] = round(dB(best_echo(1e9, 1e9, 1e10)), 1)
    num["MS_N9_ECHO_1000_N9"] = round(dB(best_echo(1e9, 1e9, 1e9)), 1)
    num["MS_PIN_1000_10"] = "%.0f pW ($%.0f$ dBm)" % (1e10 * dm[mkey(1e9, 1e9, "P_in_per_photon")] * 1e12,
                                                      10 * np.log10(1e10 * dm[mkey(1e9, 1e9, "P_in_per_photon")] * 1e3))
    p_lg0, N_lg0 = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    nb1 = (p_lg0.kappa / (4 * p_lg0.g)) ** 2 / eta  # S_ss = 1
    num["MS_LG_NBAR1"] = float("%.1e" % nb1)
    num["MS_LG_P1"] = round(nb1 * p_lg0.kappa * HBAR * (p_lg0.omega_s + p_lg0.Delta) / 4 * 1e3, 2)  # mW
    V20 = N / 4 / 100
    num["MS_RES_SC"] = float(np.sqrt(V20) / N)
    num["MS_NCRIT"] = float(dm[mkey(N, D, "n_crit")])
    num["MS_T1_TERM"] = float(N / (2 * 3 * 3600))
    num["MS_T1_VAR"] = float(round(N / (2 * 3 * 3600) * 0.1, -3))
    w = np.geomspace(1, 10, 100001)  # log-uniform coupling spread of one order of magnitude
    num["MS_NEFF_10"] = round(float(np.mean(w**2) ** 2 / np.mean(w**4)), 2)
    phi_st = 2 * chi_s * 1e9 * 1e-3
    num["MS_BAL_30"] = float("%.0e" % (0.14 / phi_st))
    num["MS_EPS_20DB"] = round(1e3 * np.sqrt(eta) / 100, 1)
    num["MS_EPS_30DB"] = round(1e3 * np.sqrt(eta) / 1000, 2)
    num["MS_PHI_RES"] = float("%.1e" % (8 * chi_s / kap * np.sqrt(V20)))
    num["MS_OAT_LOSS"] = float("%.0e" % (N * chi_s**2 * 1e-6 / 2))
    num["MS_DRIFT"] = float("%.1e" % (p.Gamma_SR * N**2 / 4))
    num["MS_DRIFT_TSS"] = float("%.1e" % (p.Gamma_SR * N**2 / 4 * num["MS_SC_T9"] * 1e-3))
    num["MS_SR_TIME"] = round(1e3 / (p.Gamma_SR * N), 1)  # ms, superradiant decay time 1/(Gamma_SR N)
    num["MS_SIGV"] = float("%.0e" % np.sqrt(V20))
    if "check_rows" in dm:
        cr = dm["check_rows"]
        num["MS_CHECK_T"] = int(round(cr[-1, 0] * 1e6))
        num["MS_CHECK_ERR"] = round(100 * float(np.max(np.abs(cr[:, 1] / cr[:, 2] - 1))), 1)
        num["MS_CHECK_TABLE"] = " \\\\\n".join(f"{r[0]*1e6:.0f} & {r[1]:.3e} & {r[2]:.3e}" for r in cr) + " \\\\"
        num["MS_CHECK_TABLE4"] = " \\\\\n".join(f"{r[0]*1e6:.0f} $\\mu$s & {r[1]:.3e} & {r[2]:.3e} & 0.96" for r in cr) + " \\\\"
        num["MS_ETA_COST"] = round(10 * np.log10(np.sqrt(0.5 / 0.17)), 1)
    for shape in ["gaussian", "lorentzian"]:
        for nb in [1e8, 1e9]:
            k = f"shape_{shape}_n{int(np.log10(nb))}_W"
            if k in dm:
                num[f"MS_{shape[:5].upper()}_W{int(np.log10(nb))}"] = round(dB(np.nanmax(dm[k])), 1)
        if f"shape_{shape}_C1ms" in dm:
            num[f"MS_{shape[:5].upper()}_C"] = round(float(dm[f"shape_{shape}_C1ms"]), 3)
    if "check_dephased_rows" in dm:
        cr = dm["check_dephased_rows"]
        num["MS_DEPH_VAR"] = float("%.1e" % cr[-1, 1])
        num["MS_DEPH_T"] = round(cr[-1, 0] * 1e3, 1)
        num["MS_DEPH_V30"] = float("%.1e" % (1e9 / 4 / 1000))
        from scipy.integrate import cumulative_trapezoid
        tD, DD = dm[mkey(1e9, 1e9, "t")], dm[mkey(1e9, 1e9, "D")]
        ID = cumulative_trapezoid(DD, tD, initial=0)
        num["MS_DEPH_TABLE"] = " \\\\\n".join(f"{r[0]*1e3:.1f} ms & {r[1]:.2e} & {np.interp(r[0], tD, ID):.2e} & {r[3]:.3f}" for r in cr) + " \\\\"
        num["MS_DEPH_ERR"] = round(100 * float(np.max(np.abs(cr[:, 1] / np.interp(cr[:, 0], tD, ID) - 1))), 0)
    if "twist_voigt" in dm:
        num["SC_30_VOIGT"] = round(-dB(dm["twist_voigt"][0]), 1)
        num["SC_30_GAUSS"] = round(-dB(dm["twist_gauss"][0]), 1)
        num["SC_30_LOR"] = round(-dB(dm["twist_lorentz"][0]), 1)
    # requirements table: SC point and loop-gap
    p_lg, N_lg = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
    rows = []
    for name, f in [("$\\chi_s/2\\pi$ (Hz)", lambda p, N: p.g**2 / p.Delta / TWO_PI),
                    ("$\\Gamma_{\\rm SR}N/2\\pi$ (Hz)", lambda p, N: p.Gamma_SR * N / TWO_PI),
                    ("$n_{\\rm crit}=\\Delta^2/4g^2$", lambda p, N: p.Delta**2 / (4 * p.g**2)),
                    ("$4g\\sqrt{\\eta_{\\rm d}\\bar n}/\\kappa$ at $\\bar n=10^8$, $\\eta_{\\rm d}=0.5$", lambda p, N: 4 * p.g * np.sqrt(0.5 * 1e8) / p.kappa),
                    ("$4g\\sqrt{\\eta_{\\rm d}\\bar n}/\\kappa$ at $\\bar n=10^9$, $\\eta_{\\rm d}=0.5$", lambda p, N: 4 * p.g * np.sqrt(0.5 * 1e9) / p.kappa),
                    ("time to steady state $3/\\sqrt{\\Gamma_m\\mathcal D}$, $\\bar n=10^9$ (ms)", lambda p, N: 3e3 / np.sqrt(64 * 0.5 * (p.g**2 / p.Delta)**2 * 1e9 / p.kappa * p.Gamma_SR * N**2 / 4)),
                    ("$P_{\\rm in}$, $\\bar n=10^9$ (pW)", lambda p, N: 1e9 * p.kappa * HBAR * (p.omega_s + p.Delta) / 4 * 1e12),
                    ("Stark shift per photon $2\\chi_s/2\\pi$ (Hz)", lambda p, N: 2 * p.g**2 / p.Delta / TWO_PI),
                    ("$2\\chi_s\\sqrt N/\\kappa$", lambda p, N: 2 * p.g**2 / p.Delta * np.sqrt(N) / p.kappa),
                    ("phase per spin $8\\chi_s/\\kappa$ (rad)", lambda p, N: 8 * p.g**2 / p.Delta / p.kappa)]:
        rows.append(f"{name} & {fmt_sci(f(p, N))} & {fmt_sci(f(p_lg, N_lg))}")
    num["MS_REQ_TABLE"] = " \\\\\n".join(rows) + " \\\\"

dc = load("conditional")
if dc is not None and dm is not None:
    tc = dc["t"]
    def best(key, col):
        return float(np.nanmax(dc[key][:, col])) if key in dc else np.nan
    num["MS_COND_TWIST"] = round(dB(best("voigt_n0_eta0.5", 4)), 1)
    num["MS_COND_TWIST_G"] = round(dB(best("gaussian_n0_eta0.5", 4)), 1)
    num["MS_COND_TWIST_L"] = round(dB(best("lorentzian_n0_eta0.5", 4)), 1)
    for nb in [8, 9]:
        num[f"MS_COND_MEAS{nb}"] = round(dB(best(f"voigt_n{nb}_eta0.5", 3)), 1)
        num[f"MS_COND_BEST{nb}"] = round(dB(best(f"voigt_n{nb}_eta0.5", 4)), 1)
        num[f"MS_COND_ETA1_BEST{nb}"] = round(dB(best(f"voigt_n{nb}_eta1.0", 4)), 1)
    # conditional variance against the Riccati model (Voigt, eta = 0.5)
    errs = []
    for nb in [8, 9]:
        rows = dc[f"voigt_n{nb}_eta0.5"]
        W = dm[mkey(1e10, 3e7, f"Wdirect_eta0.5_n{nb}")]
        te = dm[mkey(1e10, 3e7, "t_eval")]
        errs.append(np.max(np.abs(dB(rows[:, 3]) - dB(np.interp(rows[:, 0], te, W)))))
    num["MS_COND_ERR"] = round(float(max(errs)), 2)
    rows_t = []
    for shape, lab in [("voigt", "Voigt"), ("gaussian", "Gaussian"), ("lorentzian", "Lorentzian")]:
        for nb in [8, 9]:
            k = f"{shape}_n{nb}_eta0.5"
            if k in dc:
                rows_t.append(f"{lab} & $10^{{{nb}}}$ & 0.5 & {dB(best(shape + '_n0_eta0.5', 4)):.1f} & {dB(best(k, 3)):.1f} & {dB(best(k, 4)):.1f}")
    for eta in [0.8, 1.0]:
        for nb in [8, 9]:
            k = f"voigt_n{nb}_eta{eta}"
            if k in dc:
                rows_t.append(f"Voigt & $10^{{{nb}}}$ & {eta} & {dB(best('voigt_n0_eta0.5', 4)):.1f} & {dB(best(k, 3)):.1f} & {dB(best(k, 4)):.1f}")
    num["MS_COND_TABLE"] = " \\\\\n".join(rows_t) + " \\\\"

de = load("echo")
if de is not None:
    p0, _ = loop_gap_dispersive(1.0)
    chiN = p0.chi * de["N0s"] / TWO_PI
    taus = de["taus"]
    k01, k03, k1, k3 = [int(np.argmin(np.abs(taus - x))) for x in (1e-4, 3e-4, 1e-3, 3e-3)]
    ev, rv = de["mf_voigt_echo"], de["mf_voigt_ramsey"]
    j = int(np.argmin(ev[:, k03]))
    num["ECHO_MIN_03"] = round(float(ev[j, k03]), 2)
    num["ECHO_MIN_03_CHIN"] = round(float(chiN[j] / 1e3), 1)
    num["ECHO_MIN_03_L"] = round(float(de["mf_lorentzian_echo"][:, k03].min()), 2)
    num["ECHO_MIN_03_G"] = round(float(de["mf_gaussian_echo"][:, k03].min()), 2)
    num["ECHO_MIN_01"] = round(float(ev[:, k01].min()), 2)
    num["ECHO_20K_03"] = round(float(ev[-1, k03]), 2)
    num["ECHO_20K_01"] = round(float(ev[-1, k01]), 2)
    num["ECHO_LOW_3"] = float(np.floor(100 * float(ev[chiN <= 600][:, k3].min())) / 100)
    tf = de["tau_fine"]
    def at(key, col, tau):
        return float(de[key][int(np.argmin(np.abs(tf - tau))), col])
    for N0, tag in [(6e14, "AB"), (7e14, "AB7")]:
        key = f"fine_N0_{N0:.0e}"
        num[f"ECHO_{tag}_01"] = round(at(key, 0, 1e-4), 2)
        num[f"ECHO_{tag}_NOEM_01"] = round(at(key, 1, 1e-4), 2)
        num[f"ECHO_{tag}_RAMSEY_01"] = round(at(key, 2, 1e-4), 2)
        r = de[key][:, 0]
        num[f"ECHO_{tag}_MIN"] = round(float(r.min()), 2)
        num[f"ECHO_{tag}_MIN_TAU"] = int(round(tf[np.argmin(r)] * 1e6))
        # sequence of local extrema in the first 0.35 ms
        ext = [i for i in range(1, len(r) - 1) if tf[i] < 3.5e-4 and ((r[i] < r[i-1] and r[i] <= r[i+1]) or (r[i] > r[i-1] and r[i] >= r[i+1]))]
        num[f"ECHO_{tag}_SEQ"] = ", ".join(f"{r[i]:.2f} at {tf[i]*1e3:.3f} ms" for i in ext)
    num["ECHO_PERIOD_US"] = int(round(1e6 / (p0.chi * 6e14 / TWO_PI)))
    mf1 = np.interp(de["cum_N0s"], de["N0s"], ev[:, k1])
    num["ECHO_CUM_ERR"] = float(np.ceil(100 * np.max(np.abs(de["cum_echo_1ms"] - mf1))) / 100)
    dif = max(np.max(np.abs(de["noem_01"][:, 0] - ev[:, k01])), np.max(np.abs(de["noem_03"][:, 0] - ev[:, k03])))
    num["ECHO_NOEM_ERR"] = float(np.ceil(100 * dif) / 100)
    num["ECHO_CONV"] = ", ".join(f"{v:.3f}" for v in de["conv_echo"])
    num["ECHO_CONV_M"] = ", ".join(str(int(m)) for m in de["conv_M"])
    for N, tag in [(1e14, "14"), (6e14, "AB"), (2e15, "15")]:
        num[f"ECHO_TWIST_MIN_{tag}"] = round(np.sqrt(N) / (p0.chi * N) / 60, 0)
    rows = []
    for i in range(len(chiN)):
        n0 = de["N0s"][i]
        ex = int(np.floor(np.log10(n0)))
        rows.append(f"${n0/10**ex:.1f}\\times10^{{{ex}}}$ & {chiN[i]/1e3:.2f} & " + " & ".join(f"{ev[i,k]:.2f}" for k in (k01, k03, k1)) + f" & {de['noem_01'][i,0]:.2f} & {de['noem_03'][i,0]:.2f} & " + " & ".join(f"{rv[i,k]:.2f}" for k in (k01, k03, k1)))
    num["ECHO_TABLE"] = " \\\\\n".join(rows) + " \\\\"


# how closely the cumulant solution follows the mean field in Fig. 2(b)
_b = np.load(os.path.join(DATA, "benchmark.npz"))
_mf = np.interp(_b["cum_t"], _b["t"], _b["mf_voigt_7000"])
_dev = np.abs(_mf - _b["cum_contrast_7000"])
num["CUM_MF_DEV"] = float(np.ceil(100 * _dev.max()) / 100)
num["CUM_MF_DEV_3MS"] = float(np.ceil(100 * _dev[_b["cum_t"] <= 3e-3].max()) / 100)

save_json("numbers", num)
print(json.dumps(num, indent=1, default=str))


def fmt(v):
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5):
            e = int(np.floor(np.log10(abs(v))))
            mant = v / 10**e
            return r"$%.0f\times10^{%d}$" % (mant, e) if abs(mant - round(mant)) < 0.05 else r"$%.1f\times10^{%d}$" % (mant, e)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"
    return str(v)


tex = open(os.path.join(ROOT, "paper", "main.tex")).read()
missing = set()


def repl(mo):
    key = mo.group(1)
    if key in num and num[key] is not None:
        return fmt(num[key])
    missing.add(key)
    return mo.group(0)


filled = re.sub(r"\[\[([A-Z0-9_]+)\]\]", repl, tex)
open(os.path.join(ROOT, "paper", "main_filled.tex"), "w").write(filled)
tex = open(os.path.join(ROOT, "paper", "supplement.tex")).read()
filled = re.sub(r"\[\[([A-Z0-9_]+)\]\]", repl, tex)
open(os.path.join(ROOT, "paper", "supplement_filled.tex"), "w").write(filled)
print("missing placeholders:", sorted(missing))
