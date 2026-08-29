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
def fig_concept():
    """Schematic of the device, the protocol and the pipeline."""
    fig = plt.figure(figsize=(7.0, 2.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.2])
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((1, 2), 5, 5, fc="#dbe9f6", ec=C[0], lw=1))
    ax.text(3.5, 7.4, r"$^{171}$Yb$^{3+}$:CaWO$_4$", ha="center", fontsize=7)
    rng = np.random.default_rng(1)
    for x, y in rng.uniform([1.4, 2.4], [5.6, 6.6], size=(28, 2)):
        ax.annotate("", xy=(x, y + 0.45), xytext=(x, y - 0.45), arrowprops=dict(arrowstyle="->", lw=0.6, color=C[1]))
    ax.add_patch(plt.Rectangle((6.6, 1.5), 2.8, 6.0, fc="none", ec="0.3", lw=1.2, ls="-"))
    ax.text(8.0, 8.1, r"cavity $\kappa$, $\Delta$", ha="center", fontsize=7)
    ax.text(8.0, 4.3, r"$g$", ha="center", fontsize=8)
    ax.text(0.2, 0.6, r"$\gamma_{\rm inh}$, $T_2$, $n_{\rm th}$, $g_j$", fontsize=7)
    ax.text(0.2, 9.3, "(a)", fontweight="bold", fontsize=9)
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.2, 9.3, "(b)", fontweight="bold", fontsize=9)
    y0 = 6.5
    ax.plot([0.5, 9.5], [y0, y0], color="0.3", lw=0.8)
    for x, w, lab in [(1.0, 0.5, r"$\pi/2$"), (4.6, 0.7, r"$\pi$"), (8.4, 0.5, r"$\pi/2$")]:
        ax.add_patch(plt.Rectangle((x, y0), w, 1.6, fc=C[0], ec="none"))
        ax.text(x + w / 2, y0 + 2.0, lab, ha="center", fontsize=7)
    ax.annotate("", xy=(4.5, y0 - 0.5), xytext=(1.6, y0 - 0.5), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(3.0, y0 - 1.4, r"$t/2$", ha="center", fontsize=7)
    ax.annotate("", xy=(8.3, y0 - 0.5), xytext=(5.4, y0 - 0.5), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(6.9, y0 - 1.4, r"$t/2$", ha="center", fontsize=7)
    ax.text(5, 3.2, r"$H_{\rm eff}=\chi\,\hat J_+\hat J_- + \sum_j \frac{\delta_j}{2}\hat\sigma^z_j$", ha="center", fontsize=7)
    ax.text(5, 1.8, r"$\hat L_\downarrow=\sqrt{\Gamma_{\rm SR}(n_{\rm th}+1)}\,\hat J_-,\ \hat L_\uparrow=\sqrt{\Gamma_{\rm SR}n_{\rm th}}\,\hat J_+$", ha="center", fontsize=6.5)
    ax.text(5, 0.5, "echo twist: OAT survives, static disorder refocused", ha="center", fontsize=6.5)
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.2, 9.3, "(c)", fontweight="bold", fontsize=9)
    boxes = ["inputs:\nline shape\n$g,\\kappa,\\Delta,T$", "adiabatic\nelimination", "class-resolved\ncumulant\nequations", "$\\xi_R^2$, contrast\nreadout gain\ndesign map"]
    for k, b in enumerate(boxes):
        x = 0.2 + k * 2.5
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 3.2), 2.0, 3.6, boxstyle="round,pad=0.05", fc="#f3f3f3", ec="0.4", lw=0.7))
        ax.text(x + 1.0, 5.0, b, ha="center", va="center", fontsize=4.3)
        if k < 3:
            ax.annotate("", xy=(x + 2.5, 5.0), xytext=(x + 2.05, 5.0), arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(5, 1.6, "validated against exact master equations\nand the analytic $(\\Gamma_{\\rm SR}/\\chi)^{2/3}$ law", ha="center", fontsize=6)
    savefig(fig, "fig_concept")


if __name__ == "__main__":
    fig_validation()
    fig_benchmark()
    fig_loopgap()
    fig_scaling()
    fig_designmap()
    fig_inhomog_readout()
