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
print("missing placeholders:", sorted(missing))
