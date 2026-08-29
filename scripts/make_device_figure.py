"""Figure 1: three-dimensional device schematics and pulse sequences.

Panel (a) follows the system overview of Fukumori et al. (arXiv:2604.26909,
Fig. 1a): a metal block with a central loop (rectangular pocket) that holds the
crystal, two circular outer loops on either side connected to it by narrow gaps,
the optical beam through the crystal along the loop axis, and microwave
antennas at the outer loops.  Every number written on the figure is taken from
that paper (crystal 4.4 x 4.6 x 5 mm, 4.96 ppm 171Yb, 973 nm, kappa/2pi = 660 kHz,
Delta/2pi = 22 MHz, g/2pi = 15 mHz, V_m = 275 mm^3, base temperature < 30 mK,
spin temperature 80 mK).  Exact proportions of the block are schematic.  Panel (b) is the planar superconducting
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
        elif axis == "y":
            def P(a, t):
                return [centre[0] + r * np.cos(t), a, centre[1] + r * np.sin(t)]
            nrm = np.array([np.cos(tm), 0, np.sin(tm)])
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
def keyhole_rect_with_circle(x0, x1, z0, z1, y, cx, cz, r, n=36):
    """Front-face band (plane y = const) with a circular hole, as one polygon
    (the hole is reached through a zero-width slit from the top edge)."""
    th = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n + 1)
    pts = [[x0, y, z1], [cx, y, z1]]
    pts += [[cx + r * np.cos(t), y, cz + r * np.sin(t)] for t in th]
    pts += [[cx, y, z1], [x1, y, z1], [x1, y, z0], [x0, y, z0]]
    return pts


def loop_gap_scene(alpha_block=0.55):
    """Loop-gap resonator after Fukumori et al., Fig. 1a, drawn semi-transparent so
    that the loops, gaps, crystal and beam inside the block stay visible."""
    out = []
    X0, X1, Y0, Y1, Z0, Z1 = -17, 17, 0, 12, 0, 20
    zc = 10.0
    wx, wz = 6.0, 7.0
    ro, xo = 3.0, 12.0
    gh = 0.35
    S = Scene()
    S.box(-26, 26, -16, 18, -2.0, 0.0, PLATE, faces=["top", "front", "left", "right"])
    out.append((S, 1))
    # beam behind the block
    beam = np.array([0.85, 0.05, 0.15])
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.4, Y1 + 0.05, 30, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.85)
    out.append((S, 1.5))
    # block: far faces and inner walls (opaque-ish so that the openings read as holes)
    S = Scene()
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["back", "bottom", "left"], alpha=alpha_block)
    for xc in (-xo, xo):
        cylinder_wall(S, "y", (xc, zc), ro, Y0, Y1, 0, 2 * np.pi, CU_DARK, n=36, inner=True, alpha=0.85)
    for poly, nrm in [
        ([[-wx, Y0, zc - wz], [wx, Y0, zc - wz], [wx, Y1, zc - wz], [-wx, Y1, zc - wz]], [0, 0, 1]),
        ([[-wx, Y0, zc + wz], [wx, Y0, zc + wz], [wx, Y1, zc + wz], [-wx, Y1, zc + wz]], [0, 0, -1]),
        ([[-wx, Y0, zc - wz], [-wx, Y1, zc - wz], [-wx, Y1, zc + wz], [-wx, Y0, zc + wz]], [1, 0, 0]),
        ([[wx, Y0, zc - wz], [wx, Y1, zc - wz], [wx, Y1, zc + wz], [wx, Y0, zc + wz]], [-1, 0, 0]),
    ]:
        S.add(poly, CU_DARK, normal=nrm, alpha=0.85)
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, Y0, zc - gh], [xb, Y0, zc - gh], [xb, Y1, zc - gh], [xa, Y1, zc - gh]], CU_DARK * 0.7, normal=[0, 0, 1])
        S.add([[xa, Y0, zc + gh], [xb, Y0, zc + gh], [xb, Y1, zc + gh], [xa, Y1, zc + gh]], CU_DARK * 0.7, normal=[0, 0, -1])
    out.append((S, 2))
    # crystal (a x a x c = 4.4 x 4.6 x 5 mm, c along y)
    S = Scene()
    cy = 0.5 * (Y0 + Y1)
    S.box(-2.2, 2.2, cy - 2.5, cy + 2.5, zc - 2.3, zc + 2.3, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.5, alpha=0.45)
    out.append((S, 3))
    # beam inside the block
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.4, Y0, Y1, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.85)
    out.append((S, 3.5))
    # front, top and right faces, semi-transparent
    S = Scene()
    yf = Y0
    S.add(keyhole_rect_with_circle(X0, -wx, Z0, Z1, yf, -xo, zc, ro), CU, normal=[0, -1, 0], alpha=alpha_block)
    S.add(keyhole_rect_with_circle(wx, X1, Z0, Z1, yf, xo, zc, ro), CU, normal=[0, -1, 0], alpha=alpha_block)
    S.add([[-wx, yf, Z0], [wx, yf, Z0], [wx, yf, zc - wz], [-wx, yf, zc - wz]], CU, normal=[0, -1, 0], alpha=alpha_block)
    S.add([[-wx, yf, zc + wz], [wx, yf, zc + wz], [wx, yf, Z1], [-wx, yf, Z1]], CU, normal=[0, -1, 0], alpha=alpha_block)
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, yf - 0.02, zc - gh], [xb, yf - 0.02, zc - gh], [xb, yf - 0.02, zc + gh], [xa, yf - 0.02, zc + gh]], CU_DARK * 0.5, flat=True)
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["top", "right"], alpha=alpha_block)
    out.append((S, 4))
    # beam in front of the block
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.4, -20, Y0 - 0.05, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.85)
    out.append((S, 5))
    return out, dict(zc=zc, xo=xo, ro=ro, wx=wx, wz=wz, X0=X0, X1=X1, Y0=Y0, Y1=Y1, Z1=Z1, cy=cy)


def sc_scene():
    """Planar superconducting resonator on a thin CaWO4 chip (proposed, not built):
    niobium film patterned into a meander inductor between two capacitor pads, a
    translucent read-out volume under the inductor, and the optical read-out beam."""
    out = []
    S = Scene()
    S.box(-3, 3, -3, 3, 0, 0.5, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.4, alpha=0.35, faces=["bottom", "back", "left"])
    out.append((S, 1))
    # read-out volume under the inductor (region of near-uniform microwave field)
    S = Scene()
    S.box(-1.7, 1.7, -1.9, 1.9, 0.02, 0.5, np.array([1.0, 0.55, 0.1]), alpha=0.25)
    out.append((S, 2))
    S = Scene()
    S.box(-3, 3, -3, 3, 0, 0.5, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.4, alpha=0.35, faces=["top", "front", "right"])
    out.append((S, 3))
    S = Scene()
    t = 0.5 + 0.02
    S.box(-2.7, -1.7, -2.4, 2.4, 0.5, t, NB, faces=["top", "front", "right"])
    S.box(1.7, 2.7, -2.4, 2.4, 0.5, t, NB, faces=["top", "front", "left"])
    w = 0.14
    ys = np.linspace(-1.8, 1.8, 9)
    for k, yy in enumerate(ys):
        S.box(-1.7, 1.7, yy - w / 2, yy + w / 2, 0.5, t, NB, faces=["top", "front"])
        if k < len(ys) - 1:
            xa = 1.7 - w if k % 2 == 0 else -1.7
            S.box(xa, xa + w, yy - w / 2, ys[k + 1] + w / 2, 0.5, t, NB)
    out.append((S, 4))
    # optical read-out beam, vertical through the chip at the centre of the inductor
    S = Scene()
    cylinder_wall(S, "z", (0, 0), 0.45, -0.6, 2.6, 0, 2 * np.pi, np.array([0.85, 0.05, 0.15]), n=16, inner=False, alpha=0.35)
    out.append((S, 5))
    return out


def draw_spins_2d(fig, ax, pts3d, colour=ORANGE, size=0.008, lw=0.5, ms=4):
    for p in pts3d:
        d = fig.transFigure.inverted().transform(p2d(ax, *p))
        fig.add_artist(FancyArrowPatch((d[0], d[1] - size), (d[0], d[1] + size), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=ms, lw=lw, color=colour, zorder=25))


def marker(fig, xy, num, colour="k", r=0.011, fs=6.0):
    fig.add_artist(matplotlib.patches.Circle(xy, r, transform=fig.transFigure, fc="white", ec=colour, lw=0.7, zorder=30))
    fig.text(xy[0], xy[1], str(num), fontsize=fs, ha="center", va="center", color=colour, zorder=31, fontweight="bold")


def marker3d(fig, ax, xyz, num, offset=(0.0, 0.0), colour="k"):
    p = fig.transFigure.inverted().transform(p2d(ax, *xyz))
    q = (p[0] + offset[0], p[1] + offset[1])
    if abs(offset[0]) + abs(offset[1]) > 0.012:
        fig.add_artist(FancyArrowPatch(q, p, transform=fig.transFigure, arrowstyle="-", lw=0.6, color=colour, shrinkA=4, shrinkB=0, zorder=29))
    marker(fig, q, num, colour=colour)
    return q


def text3d(fig, ax, xyz, txt, colour="k", fontsize=6.0, ha="center", va="center", rotation=0, **kw):
    p = fig.transFigure.inverted().transform(p2d(ax, *xyz))
    fig.text(p[0], p[1], txt, fontsize=fontsize, ha=ha, va=va, color=colour, rotation=rotation, zorder=32, **kw)


# ---------------------------------------------------------------------------
def make():
    fig = plt.figure(figsize=(7.0, 3.6))

    # ---------------- (a) loop-gap resonator ----------------
    ax = fig.add_axes([-0.01, 0.26, 0.60, 0.74], projection="3d")
    ax.computed_zorder = False
    scenes, G = loop_gap_scene()
    for S, zo in scenes:
        S.draw(ax, zorder=zo)
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-112)
    ax.set_xlim(-27, 27)
    ax.set_ylim(-19, 20)
    ax.set_zlim(-3, 23)
    ax.set_box_aspect((54, 39, 26), zoom=1.6)
    ax.set_axis_off()
    fig.canvas.draw()
    zc, xo, ro, wx, wz, cy = G["zc"], G["xo"], G["ro"], G["wx"], G["wz"], G["cy"]
    Y0, Y1, X0, X1, Z1 = G["Y0"], G["Y1"], G["X0"], G["X1"], G["Z1"]
    # spins inside the crystal
    rng = np.random.default_rng(5)
    spins = [(x, y, z) for x, y, z in rng.uniform([-1.7, cy - 2.0, zc - 1.8], [1.7, cy + 2.0, zc + 1.8], size=(12, 3))]
    draw_spins_2d(fig, ax, spins, colour=CRYSTAL_EDGE, size=0.006, lw=0.5, ms=3.5)

    fig.text(0.008, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
    text3d(fig, ax, (0, Y1, Z1 + 1.5), "loop-gap microwave resonator", colour="k", fontsize=6.4, va="bottom", fontweight="bold")
    text3d(fig, ax, (0, 0.5 * (Y0 + Y1), Z1), r"$\kappa/2\pi = 660$ kHz", colour="w", fontsize=5.6, va="center")
    text3d(fig, ax, (0, -11, 0), "mixing-chamber plate, base temperature below 30 mK", colour="k", fontsize=5.8, va="center")
    text3d(fig, ax, (0, Y0, zc - wz - 0.9), "central loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (-xo - 1.5, Y0, zc - ro - 0.8), "outer loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (xo, Y0, zc - ro - 0.8), "outer loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (-(xo - ro + wx) / 2, Y0, zc - 1.4), "gap", colour="w", fontsize=5.2, va="top")
    text3d(fig, ax, ((xo - ro + wx) / 2, Y0, zc - 1.4), "gap", colour="w", fontsize=5.2, va="top")
    marker3d(fig, ax, (-1.5, Y0 - 0.5, zc - 1.5), 1, offset=(-0.02, -0.075))
    marker3d(fig, ax, (X0 + 3, Y0, 2.5), 2)
    marker3d(fig, ax, (0, -18, zc), 3, offset=(0.0, -0.04), colour=RED)
    marker3d(fig, ax, (xo, Y0 - 1, zc), 4, offset=(0.04, -0.05), colour=BLUE)
    marker3d(fig, ax, (-xo, Y0 - 1, zc), 4, offset=(-0.045, -0.03), colour=BLUE)
    marker3d(fig, ax, (14, -13, 0), 5)
    for xc, sgn in [(-xo, -1), (xo, 1)]:
        p = fig.transFigure.inverted().transform(p2d(ax, xc, Y0 - 0.5, zc))
        wavy(fig, (p[0] + sgn * 0.06, p[1] - 0.075), (p[0], p[1] - 0.008), BLUE, n=4, amp=0.004, lw=1.0)
    # process glyphs at the crystal, labelled in place
    c0 = fig.transFigure.inverted().transform(p2d(ax, 0, Y0 - 0.5, zc + 1.0))
    fig.add_artist(FancyArrowPatch((c0[0] - 0.03, c0[1] - 0.006), (c0[0] + 0.03, c0[1] + 0.006), transform=fig.transFigure,
                                   arrowstyle="<|-|>", mutation_scale=7, lw=1.1, color=ORANGE, zorder=20))
    fig.text(c0[0] + 0.036, c0[1] + 0.012, r"$\chi\hat J_+\hat J_-$", fontsize=5.8, color=ORANGE, va="bottom", ha="left", zorder=33, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.7))
    wavy(fig, (c0[0] + 0.02, c0[1] + 0.03), (c0[0] + 0.07, c0[1] + 0.08), BLUE, n=4, amp=0.004, lw=0.9)
    fig.text(c0[0] + 0.076, c0[1] + 0.086, r"$\Gamma_{\rm SR}$", fontsize=5.8, color=BLUE, va="bottom", zorder=33, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.7))
    wavy(fig, (c0[0] - 0.08, c0[1] + 0.078), (c0[0] - 0.03, c0[1] + 0.03), PINK, n=4, amp=0.004, lw=0.9)
    fig.text(c0[0] - 0.084, c0[1] + 0.082, r"$\Gamma_{\rm SR}n_{\rm th}$", fontsize=5.8, color=PINK, ha="right", va="bottom", zorder=33, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.7))
    fig.add_artist(FancyArrowPatch((c0[0] + 0.012, c0[1] - 0.02), (c0[0] + 0.012, c0[1] - 0.052), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, lw=0.9, color=GREEN, zorder=20))
    fig.text(c0[0] + 0.02, c0[1] - 0.058, r"$1/T_2$", fontsize=5.6, color=GREEN, va="top", ha="center", zorder=33)

    # ---------------- compact key (below panel a) ----------------
    kx, ky = 0.012, 0.245
    fig.text(kx, ky, "Key", fontsize=6.4, fontweight="bold", va="center")
    items = [
        (1, "k", "$^{171}$Yb$^{3+}$:CaWO$_4$ crystal, 4.4 $\\times$ 4.6 $\\times$ 5 mm, 4.96 ppm; $c$ axis along the beam"),
        (2, "k", "loop-gap resonator: central loop, two outer loops, gaps; $g/2\\pi = 15$ mHz, $V_{\\rm m} = 275$ mm$^3$"),
        (3, RED, "973 nm optical input and output: pumping into $|\\!\\downarrow\\rangle$, absorption readout"),
        (4, BLUE, "microwave antennas at the outer loops: pulses, transmission $S_{21}$"),
        (5, "k", "mixing-chamber plate; spin temperature 80 mK, $n_{\\rm th} = 0.19$"),
        (6, "k", "CaWO$_4$ chip with a Nb meander inductor between capacitor pads"),
        (7, "k", "read-out volume under the inductor (near-uniform field, coupling spread $D$)"),
    ]
    y = ky - 0.03
    for num, col, txt in items:
        marker(fig, (kx + 0.012, y), num, colour=col, r=0.0095, fs=5.4)
        fig.text(kx + 0.028, y, txt, fontsize=5.4, va="center")
        y -= 0.029

    # ---------------- (b) superconducting resonator (proposed) ----------------
    bx = fig.add_axes([0.58, 0.22, 0.42, 0.70], projection="3d")
    bx.computed_zorder = False
    for S, zo in sc_scene():
        S.draw(bx, zorder=zo)
    bx.set_proj_type("ortho")
    bx.view_init(elev=26, azim=-52)
    bx.set_xlim(-3.4, 3.4)
    bx.set_ylim(-3.4, 3.4)
    bx.set_zlim(-1.2, 2.8)
    bx.set_box_aspect((6.8, 6.8, 4.0), zoom=1.45)
    bx.set_axis_off()
    fig.canvas.draw()
    rng = np.random.default_rng(3)
    spins = [(x, y, z) for x, y, z in rng.uniform([-1.4, -1.6, 0.1], [1.4, 1.6, 0.42], size=(16, 3))]
    draw_spins_2d(fig, bx, spins, colour=ORANGE, size=0.007, lw=0.5, ms=4)
    fig.text(0.585, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.615, 0.985, "planar superconducting resonator on the same crystal\n(design-map case, proposed, not built)", fontsize=6.4, va="top", linespacing=1.15)
    text3d(fig, bx, (-2.2, 0.0, 0.55), "pad", colour="w", fontsize=5.6)
    text3d(fig, bx, (2.2, 0.0, 0.55), "pad", colour="w", fontsize=5.6)
    text3d(fig, bx, (0, 2.2, 0.55), "Nb meander inductor", colour="k", fontsize=5.6, va="bottom")
    text3d(fig, bx, (0, 0, 2.7), "optical readout", colour=RED, fontsize=5.6, va="bottom")
    text3d(fig, bx, (3.05, -0.5, 0.25), "CaWO$_4$", colour=CRYSTAL_EDGE, fontsize=5.8, ha="left", va="center")
    marker3d(fig, bx, (-2.95, -2.9, 0.3), 6, offset=(-0.028, -0.02))
    marker3d(fig, bx, (1.6, -1.9, 0.25), 7, offset=(0.035, -0.03))
    fig.text(0.79, 0.215, r"$\kappa/2\pi = 3$ to 300 kHz, $g\sqrt{N}/2\pi = 0.1$ to 5 MHz, $N \approx 10^{9}$ to $10^{11}$, 20 mK", fontsize=5.8, ha="center", va="top")

    # ---------------- (c) pulse sequences ----------------
    cx = fig.add_axes([0.66, 0.02, 0.33, 0.13])
    cx.set_xlim(-2.6, 10)
    cx.set_ylim(0, 4.4)
    cx.axis("off")
    fig.text(0.585, 0.16, "(c)", fontsize=9, fontweight="bold", va="top")
    for row, (y0, pulses, spans, name) in enumerate([
        (2.7, [(0.6, 0.35, r"$\pi/2$"), (4.35, 0.5, r"$\pi$"), (8.4, 0.35, r"$\pi/2$")], [(1.0, 4.3, r"$t/2$"), (4.9, 8.35, r"$t/2$")], "echo twist"),
        (0.6, [(0.6, 0.35, r"$\pi/2$"), (4.2, 0.35, r"$\phi$"), (8.4, 0.35, "read")], [(1.0, 4.15, r"twist, $+\chi$"), (4.6, 8.35, r"untwist, $-\chi$")], "twist-untwist"),
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
