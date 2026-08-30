"""Generate every figure of the paper and the supplement from data/*.npz."""
from common import *  # noqa
import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6, "figure.dpi": 200, "savefig.dpi": 300,
    "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold", "mathtext.fallback": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.handlelength": 1.6, "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def best_rows(r, col):
    """Keep, for each distinct value of column `col`, the row with the smallest xi^2 (column 3)."""
    out = []
    for v in np.unique(r[:, col]):
        rr = r[r[:, col] == v]
        out.append(rr[np.argmin(rr[:, 3])])
    return np.array(out)


def panel_label(ax, s, x=-0.22, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")


def layout(fig, bottom=0.40, top=0.86, wspace=0.52, right=0.985):
    """Fixed margins (tight_layout does not account for legends placed outside the axes)."""
    fig.subplots_adjust(left=0.075, right=right, top=top, bottom=bottom, wspace=wspace)


def legend_below(ax, ncol=2, dy=-0.33, **kw):
    """Legend placed under the x-axis label so that it never covers data."""
    kw.setdefault("frameon", False)
    kw.setdefault("fontsize", 6.5)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, borderaxespad=0.0, **kw)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


# ---------------------------------------------------------------------------
def fig_validation():
    d = load("validation")
    if d is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    ax = axs[0]
    ax.plot(d["a_Q"], dB(d["a_xi_exact"]), "o", ms=3, color=C[6], mfc="none", label="exact (N = 8)")
    ax.plot(d["a_Q"], dB(d["a_xi_cum"]), "-", color=C[0], label="cumulant")
    ax.set_xlabel(r"$Q=\chi N t$")
    ax.set_ylabel(r"$\xi_R^2$ (dB)")
    ax.legend(frameon=False)
    ax.set_title("8 disordered spins, all noise channels", fontsize=7)
    panel_label(ax, "(a)")
    ax = axs[1]
    for k, Nb in enumerate([20, 40, 80]):
        ax.plot(d["b_Q"], dB(d[f"b_xi_exact_N{Nb}"]), "o", ms=3, mfc="none", color=C[k])
        ax.plot(d["b_Q"], dB(d[f"b_xi_cum_N{Nb}"]), "-", color=C[k], label=f"N = {Nb}")
    ax.set_xlabel(r"$Q=\chi N t$")
    ax.set_ylabel(r"$\xi_R^2$ (dB)")
    ax.legend(frameon=False)
    ax.set_title("exact (circles) vs cumulant (lines)", fontsize=7)
    panel_label(ax, "(b)")
    ax = axs[2]
    r = d["c_ratio"]
    ax.loglog(r, d["c_xi_opt"], "o", ms=3.5, color=C[0], label="cumulant, optimum")
    A = float(d["c_prefactor_xi"])
    ax.loglog(r, A * (1 / r) ** (2 / 3), "-", color=C[0], lw=0.9, label=r"$%.2f\,(\kappa/2\Delta)^{2/3}$ (this work)" % A)
    ax.loglog(r, d["c_lewis_swan"], "--", color=C[1], lw=0.9, label=r"$3.0\,(\kappa/2\Delta)^{2/3}$ (perturbative)")
    ax.set_xlabel(r"$2\Delta/\kappa$")
    ax.set_ylabel(r"$\xi^2_{\rm opt}$")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    panel_label(ax, "(c)")
    layout(fig)
    savefig(fig, "fig_validation")


# ---------------------------------------------------------------------------
def fig_benchmark():
    d = load("benchmark")
    if d is None:
        return
    t = d["t"] * 1e3
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    ax = axs[0]
    for k, c in enumerate(d["chiN"]):
        ax.plot(t, d[f"mf_voigt_{int(c)}"], color=C[k], label=r"$\chi N_0/2\pi=%.1f$ kHz" % (c / 1e3))
    ax.axhline(np.exp(-1), color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("Ramsey time (ms)")
    ax.set_ylabel("contrast")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.02)
    legend_below(ax, ncol=2, fontsize=6)
    ax.set_title("mean field, Voigt line", fontsize=7)
    panel_label(ax, "(a)")
    ax = axs[1]
    c = 7000
    for k, (shape, lab) in enumerate([("gaussian", "Gaussian"), ("voigt", "Voigt"), ("lorentzian", "Lorentzian")]):
        ax.plot(t, d[f"mf_{shape}_{c}"], color=C[k], label=lab + " (MF)")
    ax.plot(d["cum_t"] * 1e3, d[f"cum_contrast_{c}"], "k--", lw=0.9, label="Voigt (cumulant)")
    ax.set_xlabel("Ramsey time (ms)")
    ax.set_ylabel("contrast")
    ax.set_ylim(0, 1.02)
    ax.set_title(r"$\chi N_0/2\pi = 7$ kHz", fontsize=7)
    legend_below(ax, ncol=2, fontsize=6)
    panel_label(ax, "(b)")
    ax = axs[2]
    for k, c in enumerate([2000, 4000, 7000]):
        ax.plot(d["cum_t"] * 1e3, dB(d[f"cum_varmax_{c}"]), color=C[k], label=r"$\chi N_0/2\pi=%d$ kHz" % (c / 1e3))
        ax.plot(d["cum_t"] * 1e3, dB(d[f"cum_varmin_{c}"]), color=C[k], ls="--")
    ax.set_xlabel("Ramsey time (ms)")
    ax.set_ylabel(r"variance / $(N_0/4)$ (dB)")
    ax.set_title("anti-squeezed (solid), squeezed (dashed)", fontsize=7)
    legend_below(ax, ncol=2, fontsize=6)
    ax.set_ylim(-12, 45)
    panel_label(ax, "(c)")
    layout(fig)
    savefig(fig, "fig_benchmark")


# ---------------------------------------------------------------------------
def fig_loopgap():
    d = load("loopgap")
    if d is None or "a_xi_homogeneous_echo" not in d:
        return
    t = d["t_list"] * 1e3
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    ax = axs[0]
    for k, (shape, lab) in enumerate([("homogeneous", "homogeneous"), ("gaussian", "Gaussian"), ("voigt", "Voigt"), ("lorentzian", "Lorentzian")]):
        ax.semilogx(t, dB(d[f"a_xi_{shape}_echo"]), color=C[k], label=lab)
        ax.semilogx(t, dB(d[f"a_xi_{shape}_noecho"]), color=C[k], ls=":", lw=0.9)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("interaction time (ms)")
    ax.set_ylabel(r"$\xi_R^2$ (dB)")
    ax.set_ylim(-12, 8)
    ax.set_title("echo (solid), no echo (dotted)", fontsize=7)
    legend_below(ax, ncol=2, fontsize=6)
    ax.set_ylim(-12, 10)
    panel_label(ax, "(a)")
    if "b_rows" in d:
        ax = axs[1]
        rows = d["b_rows"]
        for k, (N, Nlab) in enumerate([(6e14, r"6\times10^{14}"), (1.35e15, r"1.35\times10^{15}")]):
            for s, ls, lab in [(0, "--", "homogeneous"), (1, "-", "Voigt")]:
                m = (rows[:, 0] == N) & (rows[:, 2] == s)
                r = best_rows(rows[m], 1)
                o = np.argsort(r[:, 1])
                ax.semilogx(r[o, 1] / 1e6, dB(r[o, 3]), ls=ls, color=C[k], marker="o" if s else None, ms=2.5,
                            label=r"$N_0=%s$, %s" % (Nlab, lab))
        ax.axvline(22, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
        ax.set_ylabel(r"$\xi^2_{\rm opt}$ (dB)")
        legend_below(ax, ncol=2, fontsize=6)
        panel_label(ax, "(b)")
    if "c_rows" in d:
        ax = axs[2]
        rows = d["c_rows"]
        for s, ls, lab in [(0, "--", "homogeneous"), (1, "-", "Voigt")]:
            m = rows[:, 2] == s
            r = best_rows(rows[m], 0)
            o = np.argsort(r[:, 0])
            ax.semilogx(r[o, 0], dB(r[o, 3]), ls=ls, color=C[0], marker="o" if s else None, ms=2.5, label=lab)
        ax2 = ax.twiny()
        r = best_rows(rows[rows[:, 2] == 1], 0)
        o = np.argsort(r[:, 0])
        ax2.semilogx(r[o, 6] / 1e3, dB(r[o, 3]), alpha=0)
        ax2.set_xlabel(r"$\chi N_0/2\pi$ (kHz)", fontsize=7)
        ax.set_xlabel(r"$N_0$")
        ax.set_ylabel(r"$\xi^2_{\rm opt}$ (dB)")
        legend_below(ax, ncol=2, fontsize=6)
        panel_label(ax, "(c)")
    layout(fig)
    savefig(fig, "fig_loopgap")


# ---------------------------------------------------------------------------
def fig_scaling():
    d = load("scaling")
    if d is None:
        return
    rows = d["a_rows"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    ax = axs[0]
    for k, ratio in enumerate([100, 1000, 10000]):
        m = (rows[:, 0] == ratio) & (rows[:, 2] == 0)
        r = best_rows(rows[m], 1)
        o = np.argsort(r[:, 1])
        ax.semilogx(r[o, 1] / GAMMA_INH_HZ, dB(r[o, 3]), "o-", ms=2.5, color=C[k], label=r"$2\Delta/\kappa=%g$" % ratio)
        A = 1.43
        ax.axhline(dB(A * (1 / ratio) ** (2 / 3)), color=C[k], ls=":", lw=0.7)
    for k, (s, lab) in enumerate([(1, "Gaussian"), (2, "Lorentzian")]):
        m = (rows[:, 0] == 1000) & (rows[:, 2] == s)
        r = rows[m]
        o = np.argsort(r[:, 1])
        ax.semilogx(r[o, 1] / GAMMA_INH_HZ, dB(r[o, 3]), marker="s" if s == 1 else "^", ms=2.5, ls="--", color=C[1], label=lab + r", $2\Delta/\kappa=10^3$", mfc="none")
    ax.set_xlabel(r"$\chi N/\gamma_{\rm inh}$")
    ax.set_ylabel(r"$\xi^2_{\rm opt}$ (dB)")
    legend_below(ax, ncol=2, fontsize=6)
    panel_label(ax, "(a)")
    if "b_rows" in d:
        ax = axs[1]
        rows = d["b_rows"]
        for k, ratio in enumerate([66.7, 6000]):
            m = np.isclose(rows[:, 0], ratio)
            r = rows[m]
            o = np.argsort(r[:, 1])
            ax.plot(r[o, 1] * 1e3, dB(r[o, 3]), "o-", ms=2.5, color=C[k], label=r"$2\Delta/\kappa=%g$" % ratio)
        ax.set_xlabel("cavity bath temperature (mK)")
        ax.set_ylabel(r"$\xi^2_{\rm opt}$ (dB)")
        legend_below(ax, ncol=2, fontsize=6)
        ax2 = ax.twinx()
        Ts = np.linspace(1e-3, 0.5, 200)
        from cavsqueeze import thermal_occupation
        ax2.plot(Ts * 1e3, [thermal_occupation(TWO_PI * OMEGA_S_HZ, T) for T in Ts], color="0.5", lw=0.7, ls="--")
        ax2.set_ylabel(r"$n_{\rm th}$ (dashed)", color="0.4", fontsize=7, labelpad=2)
        panel_label(ax, "(b)")
    if "c_rows" in d:
        ax = axs[2]
        rows = d["c_rows"]
        for k, ratio in enumerate([66.7, 6000]):
            m = np.isclose(rows[:, 0], ratio)
            r = rows[m]
            o = np.argsort(r[:, 1])
            ax.semilogx(r[o, 1] * 1e3, dB(r[o, 2]), "o-", ms=2.5, color=C[k], label=r"$2\Delta/\kappa=%g$" % ratio)
        ax.set_xlabel(r"single-spin $T_2$ (ms)")
        ax.set_ylabel(r"$\xi^2_{\rm opt}$ (dB)")
        legend_below(ax, ncol=2, fontsize=6)
        panel_label(ax, "(c)")
    layout(fig, wspace=0.7, right=0.975)
    savefig(fig, "fig_scaling")


# ---------------------------------------------------------------------------
def fig_designmap():
    d = load("designmap")
    if d is None:
        return
    best = d["best"]
    kappas, gNs = d["kappas"], d["gNs"]
    Z = np.full((len(kappas), len(gNs)), np.nan)
    Tt = np.full_like(Z, np.nan)
    Dl = np.full_like(Z, np.nan)
    for row in best:
        i = np.argmin(np.abs(kappas - row[0]))
        j = np.argmin(np.abs(gNs - row[1]))
        Z[i, j] = dB(row[4])
        Tt[i, j] = row[5]
        Dl[i, j] = row[3]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, M, lab, cmap in [(axs[0], Z, r"$\xi^2_{\rm opt}$ (dB)", "viridis_r"), (axs[1], np.log10(Tt * 1e6), r"$\log_{10}$ optimal time ($\mu$s)", "magma"), (axs[2], np.log10(Dl / 1e6), r"$\log_{10}\,\Delta_{\rm opt}/2\pi$ (MHz)", "cividis")]:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(gNs)))
        ax.set_xticklabels([f"{g/1e6:g}" for g in gNs])
        ax.set_yticks(range(len(kappas)))
        ax.set_yticklabels([f"{k/1e3:g}" for k in kappas])
        ax.set_xlabel(r"$g\sqrt{N}/2\pi$ (MHz)")
        ax.set_ylabel(r"$\kappa/2\pi$ (kHz)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(lab, fontsize=7)
        cm = matplotlib.colormaps[cmap]
        vmin_, vmax_ = np.nanmin(M), np.nanmax(M)
        for i in range(len(kappas)):
            for j in range(len(gNs)):
                if np.isfinite(M[i, j]):
                    rgba = cm((M[i, j] - vmin_) / (vmax_ - vmin_ + 1e-12))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=4.2, color="k" if lum > 0.5 else "w")
    panel_label(axs[0], "(a)", x=-0.3)
    panel_label(axs[1], "(b)", x=-0.3)
    panel_label(axs[2], "(c)", x=-0.3)
    fig.tight_layout()
    savefig(fig, "fig_designmap")


# ---------------------------------------------------------------------------
def fig_inhomog_readout():
    di = load("inhomog")
    dr = load("readout")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    if di is not None:
        ax = axs[0]
        for k, D in enumerate(di["D"]):
            key = f"xi_w_{int(D)}"
            if key not in di:
                continue
            ax.semilogx(di["t"] * 1e6, dB(di[key]), color=C[k], label=f"D = {int(D)}")
            ax.semilogx(di["t"] * 1e6, dB(di[f"xi_u_{int(D)}"]), color=C[k], ls=":")
        ax.set_xlabel(r"interaction time ($\mu$s)")
        ax.set_ylabel(r"$\xi^2$ (dB)")
        ax.set_title("weighted (solid), unweighted (dotted)", fontsize=7)
        legend_below(ax, ncol=3, fontsize=6)
        ax.set_ylim(-20, 2)
        panel_label(ax, "(a)")
    if dr is not None:
        ax = axs[1]
        eps = dr["eps"]
        for k, (lab, name) in enumerate([("loopgap", r"loop-gap, $N_0=6\times10^{14}$"), ("sc_9", r"SC, $N=10^9$"), ("sc_10", r"SC, $N=10^{10}$"), ("sc_11", r"SC, $N=10^{11}$")]):
            if f"{lab}_gain_tu" not in dr:
                continue
            gtu = np.nanmax(dr[f"{lab}_gain_tu"], axis=0)
            gpl = np.nanmax(dr[f"{lab}_gain_plain"], axis=0)
            ax.semilogx(eps, dB(gtu), color=C[k], label=name)
            ax.semilogx(eps, dB(gpl), color=C[k], ls=":")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel(r"detection noise $\sigma_{\rm det}/N$")
        ax.set_ylabel("metrological gain (dB)")
        ax.set_ylim(-2, 30)
        ax.set_title("twist-untwist (solid), plain (dotted)", fontsize=7)
        legend_below(ax, ncol=2, fontsize=6)
        panel_label(ax, "(b)")
        ax = axs[2]
        # required resolution for 3 dB gain versus N
        Ns, req_tu, req_pl = [], [], []
        for lab in ["sc_9", "sc_10", "sc_11", "loopgap"]:
            if f"{lab}_gain_tu" not in dr:
                continue
            N = float(dr[f"{lab}_N"])
            gtu = np.nanmax(dr[f"{lab}_gain_tu"], axis=0)
            gpl = np.nanmax(dr[f"{lab}_gain_plain"], axis=0)
            Ns.append(N)
            req_tu.append(np.max(eps[dB(gtu) >= 3]) if np.any(dB(gtu) >= 3) else np.nan)
            req_pl.append(np.max(eps[dB(gpl) >= 3]) if np.any(dB(gpl) >= 3) else np.nan)
        o = np.argsort(Ns)
        Ns = np.array(Ns)[o]
        ax.loglog(Ns, np.array(req_tu)[o], "o-", color=C[0], label="twist-untwist")
        ax.loglog(Ns, np.array(req_pl)[o], "s:", color=C[1], label="plain squeezed readout")
        ax.loglog(Ns, 0.5 / np.sqrt(Ns), "--", color="0.5", lw=0.8, label=r"projection noise, $1/(2\sqrt{N})$")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$\sigma_{\rm det}/N$ for 3 dB gain")
        legend_below(ax, ncol=1, fontsize=6)
        panel_label(ax, "(c)")
    layout(fig)
    savefig(fig, "fig_inhomog_readout")


# ---------------------------------------------------------------------------
def fig_beyond():
    """Corrections beyond the eliminated model: elimination error, reversal cost,
    line-shape uncertainty and pulse duration."""
    de, dr, db = load("elimination"), load("reversal"), load("robustness")
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.6))
    # (a) elimination error at the optimum versus g sqrt(N) / Delta
    ax = axs[0]
    if de is not None:
        rows = de["A_rows"]
        r = rows[:, 0]
        loss = dB(rows[:, 1]) - dB(rows[:, 2])
        ax.loglog(r, loss, "o", ms=4, color=C[0], label=f"exact, N = {int(de['A_N'])}")
        if "A20_xi2_full" in de:
            xf, xe = de["A20_xi2_full"], de["A20_xi2_elim"]
            ax.loglog([0.2], [dB(xf.min()) - dB(xe.min())], "s", ms=4, mfc="none", color=C[1], label="exact, N = 20")
        rr = np.geomspace(0.03, 0.5, 20)
        ax.loglog(rr, loss[-1] * (rr / r[-1]) ** 2, "--", color="0.5", lw=0.8, label=r"$\propto (g\sqrt{N}/\Delta)^2$")
        for x, lab in [(0.017, "loop-gap"), (0.08, "SC, 12.5 MHz")]:
            ax.axvline(x, color="0.75", lw=0.6)
            ax.text(x * 1.1, 0.0036, lab, rotation=90, fontsize=5.5, va="bottom", ha="left", color="0.35")
        ax.set_xlabel(r"$g\sqrt{N}/\Delta$")
        ax.set_ylabel("squeezing lost by the\nelimination (dB)")
        ax.set_xlim(0.012, 0.6)
        ax.set_ylim(0.003, 1.0)
        legend_below(ax, ncol=1, dy=-0.46, fontsize=6)
        panel_label(ax, "(a)", x=-0.42)
    # (b) twist-untwist gain with the resonator ring-down at the reversal
    ax = axs[1]
    if dr is not None:
        for k, kap in enumerate([3000, 10000, 30000, 100000]):
            key = f"B_sc_k{kap}_gain"
            if key not in dr:
                continue
            eps = dr[f"B_sc_k{kap}_eps"]
            g = np.nanmax(dr[key], axis=0)
            ax.semilogx(eps, dB(g), color=C[k], label=r"$\kappa/2\pi$ = %g kHz" % (kap / 1e3))
            ideal = f"B_sc_k{kap}_ideal_gain"
            if ideal in dr:
                ax.semilogx(eps, dB(np.nanmax(dr[ideal], axis=0)), color=C[k], ls=":", lw=0.9)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel(r"detection noise $\sigma_{\rm det}/N$")
        ax.set_ylabel("metrological gain (dB)")
        ax.set_ylim(-2, 22)
        ax.set_title("with ring-down (solid), ideal (dotted)", fontsize=6.5)
        legend_below(ax, ncol=2, dy=-0.36, fontsize=6)
        panel_label(ax, "(b)", x=-0.34)
    # (c) loop-gap optimum versus Lorentzian fraction of the line
    ax = axs[2]
    if db is not None and "a_rows" in db:
        ra = db["a_rows"]
        for echo, ls, lab in [(1, "-", "echo twist"), (0, ":", "free twisting")]:
            rr = ra[ra[:, 1] == echo]
            rr = rr[np.argsort(rr[:, 0])]
            ax.plot(rr[:, 0], -dB(rr[:, 2]), "o" + ls, ms=3, color=C[0] if echo else C[1], label=lab)
        ax.set_xlabel("Lorentzian fraction of the line")
        ax.set_ylabel("optimum squeezing (dB)")
        ax.set_xlim(-0.03, 1.03)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        fr = db["a_frac"]
        fid = db["a_fid_1e_us"]
        sel = [0, 3, 5, 6]
        ax2.set_xticks(fr[sel])
        ax2.set_xticklabels([f"{fid[k]:.0f}" for k in sel], fontsize=6)
        ax2.set_xlabel(r"free-induction 1/e time, 5 kHz line ($\mu$s)", fontsize=6.5)
        legend_below(ax, ncol=2, dy=-0.36, fontsize=6)
        panel_label(ax, "(c)", x=-0.34, y=1.22)
    # (d) finite pulse duration
    ax = axs[3]
    if db is not None and "c_rows" in db:
        rc = db["c_rows"]
        kinds = ["ideal", "noecho", "duration", "duration_noecho", "angle", "spread"]
        p_lg, N_lg = loop_gap_dispersive(6e14, T=0.08, T2=T2_SPIN)
        p_sc = from_hz(1e6 / np.sqrt(1e10), 1e4, 30e6, T=0.02, T2=T2_SPIN)
        for dev, name, chiN, col in [(0, "loop-gap", p_lg.chi * N_lg, C[0]), (1, "SC operating point", p_sc.chi * 1e10, C[1])]:
            sub = rc[rc[:, 0] == dev]
            ideal = sub[sub[:, 1] == kinds.index("ideal")]
            dur = sub[sub[:, 1] == kinds.index("duration")]
            dur = dur[np.argsort(dur[:, 2])]
            x = np.concatenate([[0.0], dur[:, 2] * chiN])
            y = np.concatenate([-dB(ideal[:, 3]), -dB(dur[:, 3])])
            ax.plot(x, y, "o-", ms=3, color=col, label=name + ", echo")
            free = sub[sub[:, 1] == kinds.index("duration_noecho")]
            free = free[np.argsort(free[:, 2])]
            noecho = sub[sub[:, 1] == kinds.index("noecho")]
            if len(free) and len(noecho):
                xf = np.concatenate([[0.0], free[:, 2] * chiN])
                yf = np.concatenate([-dB(noecho[:, 3]), -dB(free[:, 3])])
                ax.plot(xf, yf, "s:", ms=3, mfc="none", color=col, label=name + ", free")
        ax.set_xlabel(r"pulse duration $\chi N\,\tau_{\rm p}$")
        ax.set_ylabel("optimum squeezing (dB)")
        legend_below(ax, ncol=1, dy=-0.36, fontsize=6)
        panel_label(ax, "(d)", x=-0.34)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.42, wspace=0.62)
    savefig(fig, "fig_beyond")


def fig_measure():
    """Squeezing by measurement through the resonator (data/measurement.npz)."""
    dm = load("measurement")
    if dm is None:
        return
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.6))
    Ns, Deltas, nbars = dm["Ns"], dm["Deltas"], dm["N_bars"]
    kap = 2 * np.pi * 1e4
    eta = 0.5

    def key(N, D, name):
        return f"N{int(np.log10(N))}_D{int(D / 1e6)}_{name}"

    # (a) S(t) and gain at the SC operating point versus photon number
    ax = axs[0]
    N, D = 1e10, 3e7
    t = dm[key(N, D, "t_eval")]
    g = 2 * np.pi * 1e6 / np.sqrt(N)
    for k, nb in enumerate(nbars):
        tag = f"eta{eta}_n{int(np.log10(nb))}"
        S = dm[key(N, D, f"S_{tag}")]
        W = dm[key(N, D, f"Wdirect_{tag}")]
        ax.semilogx(t * 1e3, dB(S), color=C[k], label=r"$\bar n=10^{%d}$" % int(np.log10(nb)))
        ax.semilogx(t * 1e3, dB(W), color=C[k], ls="--", lw=0.9)
        ax.axhline(dB(4 * g * np.sqrt(eta * nb) / kap), color=C[k], ls=":", lw=0.7)
    if "twist_voigt" in dm and "twist_lorentz" in dm:
        lo, hi = -dB(dm["twist_voigt"][0]), -dB(dm["twist_lorentz"][0])
        ax.axhspan(min(lo, hi), max(lo, hi), color="0.85", lw=0, label="twisting, Lorentzian to Voigt")
    ax.set_xlabel("measurement time (ms)")
    ax.set_ylabel(r"$S$ (solid), $1/\xi^2$ (dashed) (dB)")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(-1, 32)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=6)
    panel_label(ax, "(a)", x=-0.34)
    # (b) steady state of every locked point against the formula
    ax = axs[1]
    t = dm[key(1e10, 3e7, "t_eval")]
    xs, ys = [], []
    for N in Ns:
        g = 2 * np.pi * 1e6 / np.sqrt(N)
        for D in Deltas:
            if D > 1.0e8:
                continue
            for et in dm["etas"]:
                for nb in nbars:
                    tag = f"eta{et}_n{int(np.log10(nb))}"
                    kk = key(N, D, f"S_{tag}")
                    if kk not in dm:
                        continue
                    S = dm[kk]
                    if not np.all(np.isfinite(S)):
                        continue
                    if nb > 0.1 * dm[key(N, D, "n_crit")]:
                        continue
                    # steady state: read at 3 / sqrt(Gamma_m D), which must lie well before the
                    # superradiant decay of the ensemble (Gamma_SR N t < 0.1)
                    Gm, D0 = dm[key(N, D, f"Gm_{tag}")], dm[key(N, D, "D")][0]
                    tss = 3 / np.sqrt(Gm * D0)
                    if tss > 0.1 / dm[key(N, D, "GN")] or tss > t[-1]:
                        continue
                    Ct = np.interp(tss, dm[key(N, D, "t")], dm[key(N, D, "contrast")])
                    xs.append(4 * g * np.sqrt(et * nb) / kap)
                    ys.append(np.interp(tss, t, S) * Ct)
    xs, ys = np.array(xs), np.array(ys)
    ax.loglog(xs, ys, "o", ms=2.5, color=C[0], alpha=0.7)
    rr = np.array([xs.min() / 2, xs.max() * 2])
    ax.loglog(rr, rr, "-", color="0.4", lw=0.8)
    ax.set_xlabel(r"$4g\sqrt{\eta_{\rm d}\bar n}/\kappa$")
    ax.set_ylabel("steady-state $S\\,C$ (numerical)")
    ax.text(0.05, 0.9, f"{len(xs)} points\nmax. deviation {100 * np.nanmax(np.abs(ys / xs - 1)):.1f}%", transform=ax.transAxes, fontsize=6, va="top")
    panel_label(ax, "(b)", x=-0.34)
    # (c) best gain versus detuning, direct protocol
    ax = axs[2]
    for i, N in enumerate(Ns):
        for nb, mk, fill in [(1e8, "o", "none"), (1e9, "o", None)]:
            tag = f"eta{eta}_n{int(np.log10(nb))}"
            ys = []
            for D in Deltas:
                W = dm[key(N, D, f"Wdirect_{tag}")]
                ys.append(np.nanmax(W) if np.any(np.isfinite(W)) else np.nan)
            ax.semilogx(Deltas / 1e6, dB(ys), marker=mk, ms=3.5, mfc=fill if fill else C[i], color=C[i], lw=0.9,
                        label=r"$N=10^{%d}$, $\bar n=10^{%d}$" % (int(np.log10(N)), int(np.log10(nb))))
    ax.axvline(200, color="0.6", lw=0.7)
    ax.text(200, 31.5, r"$\Delta_{\rm th}$", fontsize=6, color="0.35", ha="center", va="top")
    ax.set_xlabel(r"$\Delta/2\pi$ (MHz)")
    ax.set_ylabel("best metrological gain (dB)")
    ax.set_ylim(-1, 32)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=5.5)
    panel_label(ax, "(c)", x=-0.34)
    # (d) echo protocol: contrast and gain for N = 1e9
    ax = axs[3]
    N = 1e9
    ax2 = ax.twinx()
    for k, D in enumerate([3e7, 2e8, 3e8, 1e9]):
        taus = dm[key(N, D, "taus")]
        Ce = dm[key(N, D, "C_echo")]
        ax.semilogx(taus * 1e3, Ce, "o-", ms=3, color=C[k], label=r"$\Delta/2\pi$ = %g MHz" % (D / 1e6))
        tag = f"eta{eta}_n10"
        We = dm[key(N, D, f"Wecho_{tag}")]
        ax2.semilogx(taus * 1e3, dB(We), "s:", ms=3, mfc="none", color=C[k])
    ax.set_xlabel(r"echo half-time $\tau$ (ms)")
    ax.set_ylabel("refocused contrast (circles)")
    ax2.set_ylabel(r"gain, $\bar n=10^{10}$ (dB, squares)", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(-1, 32)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=5.5)
    panel_label(ax, "(d)", x=-0.34)
    fig.subplots_adjust(left=0.085, right=0.95, top=0.9, bottom=0.42, wspace=0.7)
    savefig(fig, "fig_measure")


def fig_echo():
    """Spin echo against the cavity-mediated interaction (data/echo.npz)."""
    de = load("echo")
    if de is None:
        return
    p0, _ = loop_gap_dispersive(1.0)
    chiN = p0.chi * de["N0s"] / TWO_PI
    taus = de["taus"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    # (a) echo and no-pulse contrast at 2 tau versus chi N0, full model and without emission
    ax = axs[0]
    for k, (tau, tag) in enumerate([(1e-4, "01"), (3e-4, "03")]):
        kt = int(np.argmin(np.abs(taus - tau)))
        ax.semilogx(chiN / 1e3, de["mf_voigt_echo"][:, kt], "-", color=C[k], label=r"echo, $\tau$ = %g ms" % (tau * 1e3))
        if f"noem_{tag}" in de:
            ax.semilogx(chiN / 1e3, de[f"noem_{tag}"][:, 0], "--", color=C[k], lw=0.9, label="same, no emission")
        ax.semilogx(chiN / 1e3, de["mf_voigt_ramsey"][:, kt], ":", color=C[k], lw=0.9, label="no pulse")
    ax.axvline(GAMMA_INH_HZ / 1e3, color="0.6", lw=0.7)
    ax.text(GAMMA_INH_HZ / 1e3 * 0.92, 0.97, r"$\gamma_{\rm inh}$", fontsize=6, color="0.35", va="top", ha="right")
    ax.axvspan(6.1, 7.2, color="0.88", lw=0, zorder=0)
    ax.set_xlabel(r"interaction $\chi N_0/2\pi$ (kHz)")
    ax.set_ylabel(r"contrast at $2\tau$")
    ax.set_ylim(0, 1.08)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=5.5)
    panel_label(ax, "(a)", x=-0.3)
    # (b) fine tau scan at the operating point
    ax = axs[1]
    if "tau_fine" in de:
        tf = de["tau_fine"] * 1e3
        for k, (key, lab) in enumerate([("fine_N0_6e+14", r"$N_0=6\times10^{14}$ (6.1 kHz)"), ("fine_N0_7e+14", r"$N_0=7\times10^{14}$ (7.2 kHz)")]):
            if key in de:
                r = de[key]
                ax.plot(tf, r[:, 0], "-", color=C[k], label=lab + ", echo")
                ax.plot(tf, r[:, 1], "--", color=C[k], lw=0.8, label="no emission")
                ax.plot(tf, r[:, 2], ":", color=C[k], lw=0.9, label="no pulse")
    ax.set_xlabel(r"echo half-time $\tau$ (ms)")
    ax.set_ylabel(r"contrast at $2\tau$")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 0.6)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=5.5)
    panel_label(ax, "(b)", x=-0.3)
    # (c) line shapes and the cumulant check
    ax = axs[2]
    k03 = int(np.argmin(np.abs(taus - 3e-4)))
    k1 = int(np.argmin(np.abs(taus - 1e-3)))
    for j, (tag, lab) in enumerate([("voigt", "Voigt"), ("gaussian", "Gaussian"), ("lorentzian", "Lorentzian")]):
        ax.semilogx(chiN / 1e3, de[f"mf_{tag}_echo"][:, k03], "-", color=C[j], label=lab + r", $\tau$ = 0.3 ms")
    ax.semilogx(chiN / 1e3, de["mf_voigt_echo"][:, k1], "-", color=C[3], lw=0.9, label=r"Voigt, $\tau$ = 1 ms, mean field")
    if "cum_N0s" in de:
        ax.semilogx(p0.chi * de["cum_N0s"] / TWO_PI / 1e3, de["cum_echo_1ms"], "o", ms=4, mfc="none", color=C[3], label=r"Voigt, $\tau$ = 1 ms, cumulant")
    ax.axvline(GAMMA_INH_HZ / 1e3, color="0.6", lw=0.7)
    ax.set_xlabel(r"interaction $\chi N_0/2\pi$ (kHz)")
    ax.set_ylabel(r"echo contrast at $2\tau$")
    ax.set_ylim(0, 1.08)
    legend_below(ax, ncol=2, dy=-0.36, fontsize=5.5)
    panel_label(ax, "(c)", x=-0.3)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.42, wspace=0.5)
    savefig(fig, "fig_echo")


if __name__ == "__main__":
    fig_validation()
    fig_benchmark()
    fig_loopgap()
    fig_scaling()
    fig_designmap()
    fig_inhomog_readout()
    fig_beyond()
    fig_measure()
    fig_echo()
