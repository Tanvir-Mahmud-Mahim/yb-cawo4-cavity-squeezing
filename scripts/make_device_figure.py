"""Figure 1: three-dimensional device schematics and pulse sequences.

Panel (a) is a cutaway of the loop-gap resonator experiment of Fukumori et al.
(arXiv:2604.26909): every dimension and parameter written on the figure is taken
from that paper (crystal 4.4 x 4.6 x 5 mm, 4.96 ppm 171Yb, 973 nm optical beam,
kappa/2pi = 660 kHz, Delta/2pi = 22 MHz, g/2pi = 15 mHz, V_m = 275 mm^3,
mixing-chamber base temperature < 30 mK, spin temperature 80 mK).  The shape of
the resonator body is schematic.  Panel (b) is the planar superconducting
resonator considered in the design map (a proposal, not a fabricated device).
Panel (c) shows the two pulse sequences used in the paper.
"""
from common import *  # noqa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

plt.rcParams.update({
    "font.size": 7, "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold", "mathtext.fallback": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400, "figure.dpi": 200,
})

# colours
CU = np.array([0.80, 0.52, 0.28])       # copper body
CU_DARK = CU * 0.62
CU_LIGHT = np.clip(CU * 1.18, 0, 1)
PLATE = np.array([0.85, 0.72, 0.36])    # gold-plated plate
CRYSTAL = (0.60, 0.78, 0.93)
CRYSTAL_EDGE = (0.10, 0.35, 0.60)
NB = np.array([0.30, 0.32, 0.38])       # niobium film
RED = "#c8102e"
ORANGE = "#e06a00"
BLUE = "#1f6fb2"
PINK = "#c0509a"
GREEN = "#1a9c5a"
GREY = "#5a5a5a"

LIGHT = np.array([-0.45, -0.6, 0.66])
LIGHT /= np.linalg.norm(LIGHT)


def shade(base, normal, amb=0.55, diff=0.55):
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    k = amb + diff * max(0.0, float(np.dot(n, LIGHT)))
    return tuple(np.clip(np.asarray(base) * k, 0, 1))


def quad_normal(poly):
    p = np.asarray(poly)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    return n / (np.linalg.norm(n) + 1e-12)


class Scene:
    """Collect polygons (with colours) and draw them as one depth-sorted collection."""

    def __init__(self):
        self.polys, self.fc, self.ec, self.lw, self.alpha = [], [], [], [], []

    def add(self, poly, colour, edge="none", lw=0.0, alpha=1.0, normal=None, flat=None):
        poly = np.asarray(poly, float)
        if flat is None:
            n = normal if normal is not None else quad_normal(poly)
            col = shade(colour, n)
        else:
            col = tuple(colour)
        self.polys.append(poly)
        self.fc.append((*col, alpha))
        self.ec.append(edge if edge != "none" else (0, 0, 0, 0))
        self.lw.append(lw)

    def box(self, x0, x1, y0, y1, z0, z1, colour, edge="none", lw=0.0, alpha=1.0, faces=None):
        c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        c = np.array(c)
        F = {
            "bottom": ([0, 3, 2, 1], [0, 0, -1]), "top": ([4, 5, 6, 7], [0, 0, 1]),
            "front": ([0, 1, 5, 4], [0, -1, 0]), "back": ([2, 3, 7, 6], [0, 1, 0]),
            "left": ([0, 4, 7, 3], [-1, 0, 0]), "right": ([1, 2, 6, 5], [1, 0, 0]),
        }
        for name, (idx, n) in F.items():
            if faces is None or name in faces:
                self.add(c[idx], colour, edge=edge, lw=lw, alpha=alpha, normal=n)

    def draw(self, ax, zorder=1):
        coll = Poly3DCollection(self.polys, facecolors=self.fc, edgecolors=self.ec, linewidths=self.lw, zsort="average")
        coll.set_zorder(zorder)
        ax.add_collection3d(coll)
        return coll


def cylinder_wall(scene, axis, centre, r, a0, a1, th0, th1, colour, n=28, inner=True, alpha=1.0):
    """Wall of a cylinder whose axis is along `axis` ('x' or 'z'); the surface spans
    axial coordinate a0..a1 and polar angle th0..th1. `inner` flips the normal."""
    th = np.linspace(th0, th1, n + 1)
    for k in range(n):
        t0, t1 = th[k], th[k + 1]
        tm = 0.5 * (t0 + t1)
        if axis == "x":
            def P(a, t):
                return [a, centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)]
            nrm = np.array([0, np.cos(tm), np.sin(tm)])
        else:
            def P(a, t):
                return [centre[0] + r * np.cos(t), centre[1] + r * np.sin(t), a]
            nrm = np.array([np.cos(tm), np.sin(tm), 0])
        if inner:
            nrm = -nrm
        scene.add([P(a0, t0), P(a1, t0), P(a1, t1), P(a0, t1)], colour, normal=nrm, alpha=alpha)


def disc(scene, axis, centre, r, a, colour, th0=0, th1=2 * np.pi, n=40, normal_sign=1, alpha=1.0, edge="none", lw=0):
    th = np.linspace(th0, th1, n + 1)
    if axis == "x":
        pts = [[a, centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)] for t in th]
        nrm = [normal_sign, 0, 0]
    else:
        pts = [[centre[0] + r * np.cos(t), centre[1] + r * np.sin(t), a] for t in th]
        nrm = [0, 0, normal_sign]
    scene.add(pts, colour, normal=nrm, alpha=alpha, edge=edge, lw=lw)


def p2d(ax, x, y, z):
    """Project a 3-D point of `ax` to display-independent 2-D axes coordinates."""
    xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return ax.transData.transform((xs, ys))


def label3d(fig, ax, xyz, text, xytext, colour="k", fontsize=6.5, ha="left", va="center", arrow=True, lw=0.6, style="-|>", rad=0.0):
    """Annotation from a 3-D point to a 2-D figure-fraction text position."""
    disp = p2d(ax, *xyz)
    figc = fig.transFigure.inverted().transform(disp)
    kw = dict(arrowstyle=style, lw=lw, color=colour, shrinkA=0, shrinkB=1, connectionstyle=f"arc3,rad={rad}") if arrow else None
    fig.text(xytext[0], xytext[1], text, fontsize=fontsize, ha=ha, va=va, color=colour)
    if arrow:
        fig.add_artist(FancyArrowPatch(xytext, figc, transform=fig.transFigure, **kw))


def wavy(fig, p0, p1, colour, n=7, amp=0.006, lw=1.0, arrow=True):
    """A wavy (photon) line between two figure-fraction points."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = p1 - p0
    L = np.linalg.norm(d)
    u = d / L
    v = np.array([-u[1], u[0]])
    s = np.linspace(0, 1, 200)
    pts = p0[None, :] + s[:, None] * d[None, :] + (amp * np.sin(2 * np.pi * n * s))[:, None] * v[None, :]
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D(pts[:, 0], pts[:, 1], transform=fig.transFigure, color=colour, lw=lw, solid_capstyle="round"))
    if arrow:
        fig.add_artist(FancyArrowPatch(pts[-8], pts[-1], transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=lw, color=colour))


# ---------------------------------------------------------------------------
def loop_gap_scene():
    """Cutaway of the loop-gap resonator (front half, y < 0, removed).
    Returns a list of (scene, zorder) drawn back to front."""
    out = []
    X0, X1, Y0, Y1, Z0, Z1 = -12, 12, 0, 8, 0, 26
    rc, ro, g = 3.6, 2.2, 0.5          # central loop, outer loops, gap width (mm)
    zc, zo1, zo2 = 13.0, 4.0, 22.0     # loop centres along z (bore axis along x)
    # 1) plate
    S = Scene()
    S.box(-24, 24, -14, 16, -2.0, 0.0, PLATE, faces=["top", "front", "left", "right"])
    out.append((S, 1))
    # 2) resonator body: far parts first (back, bottom, left end, inner walls)
    S = Scene()
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["back", "bottom"])
    Snear = Scene()
    Snear.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["top"])
    for xe, sign in [(X0, -1), (X1, 1)]:
        th = np.linspace(np.pi / 2, 3 * np.pi / 2, 24)
        pts = [[xe, Y0, Z0], [xe, Y0, zo1 - ro]]
        pts += [[xe, ro * np.cos(t) * -1, zo1 + ro * np.sin(t)] for t in th[::-1]]
        pts += [[xe, Y0, zo1 + ro], [xe, Y0, zc - rc]]
        pts += [[xe, rc * np.cos(t) * -1, zc + rc * np.sin(t)] for t in th[::-1]]
        pts += [[xe, Y0, zc + rc], [xe, Y0, zo2 - ro]]
        pts += [[xe, ro * np.cos(t) * -1, zo2 + ro * np.sin(t)] for t in th[::-1]]
        pts += [[xe, Y0, zo2 + ro], [xe, Y0, Z1], [xe, Y1, Z1], [xe, Y1, Z0]]
        (S if sign < 0 else Snear).add(pts, CU, normal=[sign, 0, 0])
    # inner walls of the three loops (half cylinders on the y >= 0 side)
    for (zz, rr) in [(zc, rc), (zo1, ro), (zo2, ro)]:
        cylinder_wall(S, "x", (0, zz), rr, X0, X1, 0, np.pi, CU_DARK, n=30, inner=True)
    out.append((S, 2))
    # near parts: cut face (y = 0) between the openings with the two gap slots, top, right end, ports
    S = Snear
    pieces = [(Z0, zo1 - ro), (zo1 + ro, zc - rc), (zc + rc, zo2 - ro), (zo2 + ro, Z1)]
    for (za, zb) in pieces:
        zg = 0.5 * (za + zb)
        if (za, zb) in [pieces[1], pieces[2]]:
            for (a, b) in [(za, zg - g / 2), (zg + g / 2, zb)]:
                S.add([[X0, 0, a], [X1, 0, a], [X1, 0, b], [X0, 0, b]], CU_LIGHT, normal=[0, -1, 0])
            S.add([[X0, g, zg - g / 2], [X1, g, zg - g / 2], [X1, g, zg + g / 2], [X0, g, zg + g / 2]], CU_DARK * 0.8, normal=[0, -1, 0])
        else:
            S.add([[X0, 0, za], [X1, 0, za], [X1, 0, zb], [X0, 0, zb]], CU_LIGHT, normal=[0, -1, 0])
    # SMA ports on the top face feeding the top outer loop (placement illustrative)
    for xx in [-6.0, 6.0]:
        cylinder_wall(S, "z", (xx, 4.0), 1.3, Z1, Z1 + 2.6, 0, 2 * np.pi, np.array([0.75, 0.75, 0.78]), n=18, inner=False)
        disc(S, "z", (xx, 4.0), 1.3, Z1 + 2.6, np.array([0.80, 0.80, 0.83]), normal_sign=1)
        cylinder_wall(S, "z", (xx, 4.0), 0.5, Z1 + 2.6, Z1 + 5.0, 0, 2 * np.pi, np.array([0.55, 0.55, 0.58]), n=12, inner=False)
    out.append((S, 2.5))
    # 3) crystal (a x a x c = 4.4 x 4.6 x 5 mm; c-axis along the bore)
    S = Scene()
    cz = zc
    S.box(-2.5, 2.5, -2.3, 2.3, cz - 2.2, cz + 2.2, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.5, alpha=0.55)
    out.append((S, 3))
    # 4) beam and collimators
    S = Scene()
    cylinder_wall(S, "x", (0, cz), 0.35, -30, 30, 0, 2 * np.pi, np.array([0.85, 0.05, 0.15]), n=14, inner=False, alpha=0.9)
    for xa, xb in [(-34, -30), (30, 34)]:
        cylinder_wall(S, "x", (0, cz), 1.6, xa, xb, 0, 2 * np.pi, np.array([0.55, 0.55, 0.58]), n=18, inner=False)
        disc(S, "x", (0, cz), 1.6, xa if xa < 0 else xb, np.array([0.55, 0.55, 0.58]), normal_sign=-1 if xa < 0 else 1)
    out.append((S, 4))
    return out, dict(zc=zc, zo1=zo1, zo2=zo2, rc=rc, ro=ro, X0=X0, X1=X1, Y1=Y1, Z1=Z1)


def sc_scene():
    """Planar superconducting resonator on a thin CaWO4 chip (proposed)."""
    S = Scene()
    # chip 6 x 6 x 0.5 mm scaled: x,y in [-3,3], z in [0,0.5]
    S.box(-3, 3, -3, 3, 0, 0.5, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.4, alpha=0.75, faces=["top", "front", "right", "left", "back"])
    # gold back reflector under the chip
    S.box(-3, 3, -3, 3, -0.06, 0, PLATE, faces=["front", "right"])
    # niobium film: two capacitor pads and a meander inductor between them
    t = 0.5 + 0.02
    S.box(-2.7, -1.7, -2.4, 2.4, 0.5, t, NB, faces=["top", "front", "right"])
    S.box(1.7, 2.7, -2.4, 2.4, 0.5, t, NB, faces=["top", "front", "left"])
    w = 0.14
    ys = np.linspace(-1.8, 1.8, 9)
    for k, yy in enumerate(ys):
        S.box(-1.7, 1.7, yy - w / 2, yy + w / 2, 0.5, t, NB, faces=["top", "front"])
        if k < len(ys) - 1:
            xa = 1.7 - w if k % 2 == 0 else -1.7
            S.box(xa, xa + w, yy, ys[k + 1], 0.5, t, NB, faces=["top", "front"])
    # spins inside the chip below the inductor (arrow glyphs are added in 2-D)
    return S


def draw_spins_2d(fig, ax, pts3d, colour=ORANGE, size=0.008):
    """Small up-arrows at 3-D positions, drawn in figure coordinates."""
    for p in pts3d:
        d = fig.transFigure.inverted().transform(p2d(ax, *p))
        fig.add_artist(FancyArrowPatch((d[0], d[1] - size), (d[0], d[1] + size), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=4, lw=0.5, color=colour))


# ---------------------------------------------------------------------------
def make():
    fig = plt.figure(figsize=(7.0, 5.6))

    # ---------------- (a) loop-gap resonator ----------------
    ax = fig.add_axes([-0.03, 0.27, 0.62, 0.70], projection="3d")
    ax.computed_zorder = False
    scenes, G = loop_gap_scene()
    for S, zo in scenes:
        S.draw(ax, zorder=zo)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-118)
    ax.set_xlim(-24, 24)
    ax.set_ylim(-16, 20)
    ax.set_zlim(-4, 32)
    ax.set_box_aspect((48, 36, 36))
    ax.set_axis_off()
    fig.canvas.draw()

    zc = G["zc"]
    c0 = fig.transFigure.inverted().transform(p2d(ax, 0, -2.3, zc))
    # process glyphs at the crystal
    fig.add_artist(FancyArrowPatch((c0[0] - 0.035, c0[1] - 0.010), (c0[0] + 0.035, c0[1] + 0.010), transform=fig.transFigure,
                                   arrowstyle="<|-|>", mutation_scale=8, lw=1.2, color=ORANGE, zorder=20))
    wavy(fig, (c0[0] + 0.03, c0[1] + 0.035), (c0[0] + 0.115, c0[1] + 0.105), BLUE, n=5, lw=1.0)
    wavy(fig, (c0[0] - 0.125, c0[1] + 0.10), (c0[0] - 0.04, c0[1] + 0.035), PINK, n=5, lw=1.0)
    fig.add_artist(FancyArrowPatch((c0[0], c0[1] - 0.03), (c0[0], c0[1] - 0.075), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=1.0, color=GREEN, zorder=20))

    fig.text(0.012, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
    label3d(fig, ax, (17, 0, zc + 0.3), "973 nm pump and readout beam between two fibre\ncollimators (beam covers about 1% of the crystal face)",
            (0.30, 0.965), colour=RED, ha="left", va="center", fontsize=6.2)
    label3d(fig, ax, (0, -2.3, zc + 2.2), r"$^{171}$Yb$^{3+}$:CaWO$_4$ crystal, 4.4 $\times$ 4.6 $\times$ 5 mm",
            (0.24, 0.905), colour=CRYSTAL_EDGE, ha="left", va="center", fontsize=6.4)
    label3d(fig, ax, (-11, 4, G["Z1"]), "loop-gap microwave resonator\n(front half cut away)", (0.02, 0.80), colour="k", fontsize=6.4)
    label3d(fig, ax, (6.0, 4.0, G["Z1"] + 5.0), "microwave ports: spin drive and\ntransmission $S_{21}$ (calibrates $N_0$)", (0.40, 0.84), colour="k", fontsize=6.2)
    label3d(fig, ax, (8, 2.5, G["zo2"] + 1.8), "outer loop", (0.50, 0.72), colour="k", fontsize=6.2)
    label3d(fig, ax, (11.5, 0.3, zc - 6.0), "gap", (0.50, 0.50), colour="k", fontsize=6.2)
    label3d(fig, ax, (8, 2.5, G["zo1"] - 1.6), "outer loop", (0.50, 0.42), colour="k", fontsize=6.2)
    label3d(fig, ax, (-12, 0.0, zc - 3.4), "central loop\n(crystal glued at its centre)", (0.015, 0.47), colour="k", fontsize=6.2)
    label3d(fig, ax, (22, -12, -1), "mixing-chamber plate, base temperature below 30 mK\n(measured spin temperature 80 mK)",
            (0.34, 0.285), colour="k", fontsize=6.2, ha="left")

    # ---------------- legend box (top right) ----------------
    lx, ly, lw_, lh = 0.615, 0.485, 0.375, 0.50
    fig.add_artist(FancyBboxPatch((lx, ly), lw_, lh, boxstyle="round,pad=0.006", transform=fig.transFigure, fc="#f5f5f5", ec="0.5", lw=0.6))
    y = ly + lh - 0.022
    fig.text(lx + 0.012, y, "Loop-gap device (as built):", fontsize=6.6, va="center", fontweight="bold")
    entries = [
        (CU, "square", "loop-gap resonator, $\\kappa/2\\pi = 660$ kHz"),
        (CRYSTAL, "square", "$^{171}$Yb$^{3+}$:CaWO$_4$, 4.96 ppm $^{171}$Yb, $c$-cut faces"),
        ("#d8101f", "line", "973 nm light: pumping into $|\\!\\downarrow\\rangle$, absorption readout"),
        ((0.75, 0.75, 0.78), "square", "microwave ports and fibre collimators"),
        (PLATE, "square", "mixing-chamber plate of a dilution refrigerator"),
    ]
    y -= 0.036
    for col, kind, txt in entries:
        if kind == "square":
            fig.add_artist(Rectangle((lx + 0.015, y - 0.010), 0.022, 0.020, transform=fig.transFigure, fc=col, ec="0.3", lw=0.4))
        else:
            fig.add_artist(FancyArrowPatch((lx + 0.015, y), (lx + 0.037, y), transform=fig.transFigure, arrowstyle="-", lw=1.5, color=col))
        fig.text(lx + 0.046, y, txt, fontsize=6.1, va="center")
        y -= 0.033
    y -= 0.004
    fig.text(lx + 0.012, y, "Parameters:", fontsize=6.6, va="center", fontweight="bold")
    y -= 0.030
    pars = ["$\\omega_s/2\\pi = 3.084$ GHz, $\\gamma_{\\rm inh}/2\\pi < 5$ kHz, $T_2 > 150$ ms",
            "$g/2\\pi = 15$ mHz, $V_{\\rm m} = 275$ mm$^3$ (simulated), $N_0 \\leq 7\\times10^{14}$",
            "$\\Delta/2\\pi = 22$ MHz, $2\\Delta/\\kappa = 67$, $n_{\\rm th} = 0.19$ at 80 mK",
            "model values: $\\gamma_{\\rm inh}/2\\pi = 5$ kHz, $T_2 = 150$ ms"]
    for txt in pars:
        fig.text(lx + 0.015, y, txt, fontsize=6.1, va="center")
        y -= 0.027
    y -= 0.002
    fig.text(lx + 0.012, y, "Processes in the model:", fontsize=6.6, va="center", fontweight="bold")
    y -= 0.032
    procs = [
        (ORANGE, "<|-|>", "cavity-mediated one-axis twisting, $\\chi\\,\\hat J_+\\hat J_-$"),
        (BLUE, "wave", "collective emission through the cavity, $\\Gamma_{\\rm SR}$"),
        (PINK, "wave", "thermal cavity photons, rate $\\Gamma_{\\rm SR}\\,n_{\\rm th}$"),
        (GREEN, "-|>", "single-spin dephasing, $1/T_2$"),
    ]
    for col, kind, txt in procs:
        if kind == "wave":
            wavy(fig, (lx + 0.015, y), (lx + 0.040, y), col, n=3, amp=0.004, lw=1.0)
        else:
            fig.add_artist(FancyArrowPatch((lx + 0.015, y), (lx + 0.040, y), transform=fig.transFigure, arrowstyle=kind, mutation_scale=7, lw=1.0, color=col))
        fig.text(lx + 0.046, y, txt, fontsize=6.1, va="center")
        y -= 0.032

    # ---------------- level-structure inset (bottom left) ----------------
    ix = fig.add_axes([0.015, 0.015, 0.30, 0.235])
    ix.set_xlim(0, 10)
    ix.set_ylim(0, 10)
    ix.axis("off")
    ix.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7, boxstyle="round,pad=0.1", fc="white", ec="0.5", lw=0.6))
    ix.text(5, 9.2, "energy levels of one $^{171}$Yb$^{3+}$ ion used here", ha="center", va="center", fontsize=6.3)
    ix.plot([1.6, 4.4], [1.8, 1.8], color="k", lw=1.2)
    ix.plot([1.6, 4.4], [3.4, 3.4], color="k", lw=1.2)
    ix.text(4.6, 1.8, r"$|\!\downarrow\rangle$", va="center", fontsize=7)
    ix.text(4.6, 3.4, r"$|\!\uparrow\rangle$", va="center", fontsize=7)
    ix.annotate("", xy=(3.4, 3.35), xytext=(3.4, 1.85), arrowprops=dict(arrowstyle="<->", lw=0.6, color=BLUE))
    ix.text(1.45, 2.6, "3.084 GHz\nzero-field\nclock\ntransition", fontsize=5.2, va="center", ha="right", color=BLUE)
    ix.plot([1.6, 4.4], [7.6, 7.6], color="k", lw=1.2)
    ix.text(4.6, 7.6, r"$|e\rangle$", va="center", fontsize=7)
    ix.annotate("", xy=(2.2, 7.55), xytext=(2.2, 1.85), arrowprops=dict(arrowstyle="<->", lw=0.7, color=RED))
    ix.annotate("", xy=(2.9, 7.55), xytext=(2.9, 3.45), arrowprops=dict(arrowstyle="<->", lw=0.7, color=RED))
    ix.text(1.3, 5.8, "973 nm\nA, E", fontsize=5.4, color=RED, ha="right", va="center")
    ix.text(7.3, 5.6, "optical transitions A and E:\ninitialise the spins in " + r"$|\!\downarrow\rangle$" + "\nand read the " + r"$|\!\uparrow\rangle$, $|\!\downarrow\rangle$" + "\npopulations by absorption", fontsize=5.2, ha="center", va="center")
    ix.text(7.3, 2.3, r"microwave $\pi/2$ and $\pi$" + "\npulses applied through\nthe resonator", fontsize=5.2, ha="center", va="center")

    # ---------------- (b) superconducting resonator (proposed) ----------------
    bx = fig.add_axes([0.60, 0.13, 0.40, 0.34], projection="3d")
    bx.computed_zorder = False
    S2 = sc_scene()
    S2.draw(bx)
    bx.set_proj_type("ortho")
    bx.view_init(elev=28, azim=-55)
    bx.set_xlim(-3.5, 3.5)
    bx.set_ylim(-3.5, 3.5)
    bx.set_zlim(-1.5, 2.5)
    bx.set_box_aspect((7, 7, 4))
    bx.set_axis_off()
    fig.canvas.draw()
    rng = np.random.default_rng(3)
    spins = [(x, y, 0.32) for x, y in rng.uniform([-1.4, -1.6], [1.4, 1.6], size=(14, 2))]
    draw_spins_2d(fig, bx, spins, colour=ORANGE, size=0.006)
    sp = fig.transFigure.inverted().transform(p2d(bx, 0, 0, 0.55))
    fig.add_artist(matplotlib.patches.Ellipse(sp, 0.05, 0.02, transform=fig.transFigure, fc=(0.85, 0.05, 0.15, 0.35), ec="none", zorder=20))
    fig.text(0.612, 0.47, "(b)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.645, 0.468, "planar superconducting resonator on the same\ncrystal: the design-map case (proposed, not built)", fontsize=6.2, va="top")
    label3d(fig, bx, (0.0, 0.0, 0.52), "Nb meander inductor; spins under the\nnear-uniform field region are read out\noptically (coupling spread $D$)", (0.63, 0.20), colour="k", fontsize=6.0, va="top")
    label3d(fig, bx, (-2.2, 0.0, 0.52), "capacitor pads", (0.62, 0.395), colour="k", fontsize=6.0)
    label3d(fig, bx, (3.0, -2.0, 0.25), r"CaWO$_4$ chip, $N \approx 10^{10}$ spins", (0.83, 0.20), colour=CRYSTAL_EDGE, fontsize=6.0, va="top")
    fig.text(0.985, 0.405, r"$\kappa/2\pi = 3$ to 300 kHz" + "\n" + r"$g\sqrt{N}/2\pi = 0.1$ to 5 MHz" + "\n20 mK, $T_2 = 150$ ms", fontsize=6.0, ha="right", va="top")

    # ---------------- (c) pulse sequences ----------------
    cx = fig.add_axes([0.635, 0.012, 0.355, 0.12])
    cx.set_xlim(-2.6, 10)
    cx.set_ylim(0, 4.4)
    cx.axis("off")
    fig.text(0.612, 0.135, "(c)", fontsize=9, fontweight="bold", va="top")
    for row, (y0, pulses, spans, name) in enumerate([
        (2.7, [(0.6, 0.35, r"$\pi/2$"), (4.35, 0.5, r"$\pi$"), (8.4, 0.35, r"$\pi/2$")], [(1.0, 4.3, r"$t/2$"), (4.9, 8.35, r"$t/2$")], "echo twist"),
        (0.6, [(0.6, 0.35, r"$\pi/2$"), (4.2, 0.35, r"$\phi$"), (8.4, 0.35, "read")], [(1.0, 4.15, r"twist, $+\chi$"), (4.6, 8.35, r"untwist, $-\chi$")], "twist-untwist readout"),
    ]):
        cx.plot([0.3, 9.9], [y0, y0], color="0.3", lw=0.7)
        for x, w, lab in pulses:
            cx.add_patch(Rectangle((x, y0), w, 0.8, fc=BLUE, ec="none"))
            cx.text(x + w / 2, y0 + 0.9, lab, ha="center", va="bottom", fontsize=5.8)
        for xa, xb, lab in spans:
            cx.annotate("", xy=(xb, y0 - 0.2), xytext=(xa, y0 - 0.2), arrowprops=dict(arrowstyle="<->", lw=0.5))
            cx.text(0.5 * (xa + xb), y0 - 0.28, lab, ha="center", va="top", fontsize=5.6)
        cx.text(0.1, y0 + 0.4, name, ha="right", va="center", fontsize=5.8, style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_device.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("figure fig_device")


if __name__ == "__main__":
    make()
