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


def loop_gap_scene():
    """Loop-gap resonator after Fukumori et al., Fig. 1a: block with a central
    rectangular loop (crystal pocket), two circular outer loops and thin gaps,
    all running through the block along y.  Returns [(scene, zorder), ...]."""
    out = []
    X0, X1, Y0, Y1, Z0, Z1 = -17, 17, 0, 12, 0, 20   # block (mm, schematic)
    zc = 10.0                                         # loop axis height
    wx, wz = 6.0, 7.0                                 # central pocket half-widths (x, z)
    ro, xo = 3.0, 12.0                                # outer loops: radius, |x| of centre
    gh = 0.35                                         # gap half-height
    # 1) plate
    S = Scene()
    S.box(-26, 26, -16, 18, -2.0, 0.0, PLATE, faces=["top", "front", "left", "right"])
    out.append((S, 1))
    # 2) block far parts: back, bottom, left end, inner walls of the openings
    S = Scene()
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["back", "bottom", "left"])
    for xc in (-xo, xo):
        cylinder_wall(S, "y", (xc, zc), ro, Y0, Y1, 0, 2 * np.pi, CU_DARK, n=36, inner=True)
    # walls of the rectangular pocket
    for poly, nrm in [
        ([[-wx, Y0, zc - wz], [wx, Y0, zc - wz], [wx, Y1, zc - wz], [-wx, Y1, zc - wz]], [0, 0, 1]),   # floor
        ([[-wx, Y0, zc + wz], [wx, Y0, zc + wz], [wx, Y1, zc + wz], [-wx, Y1, zc + wz]], [0, 0, -1]),  # ceiling
        ([[-wx, Y0, zc - wz], [-wx, Y1, zc - wz], [-wx, Y1, zc + wz], [-wx, Y0, zc + wz]], [1, 0, 0]),  # left wall
        ([[wx, Y0, zc - wz], [wx, Y1, zc - wz], [wx, Y1, zc + wz], [wx, Y0, zc + wz]], [-1, 0, 0]),     # right wall
    ]:
        S.add(poly, CU_DARK, normal=nrm)
    # gap slots: thin channels through the block between the pocket and each loop
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, Y0, zc - gh], [xb, Y0, zc - gh], [xb, Y1, zc - gh], [xa, Y1, zc - gh]], CU_DARK * 0.7, normal=[0, 0, 1])
        S.add([[xa, Y0, zc + gh], [xb, Y0, zc + gh], [xb, Y1, zc + gh], [xa, Y1, zc + gh]], CU_DARK * 0.7, normal=[0, 0, -1])
    out.append((S, 2))
    # 3) crystal inside the pocket: a x a x c = 4.4 x 4.6 x 5 mm, c axis along y (beam)
    S = Scene()
    cy = 0.5 * (Y0 + Y1)
    S.box(-2.2, 2.2, cy - 2.5, cy + 2.5, zc - 2.3, zc + 2.3, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.5, alpha=0.6)
    out.append((S, 3))
    # 4) front face with the three openings and the gap slits, top and right faces
    S = Scene()
    yf = Y0
    S.add(keyhole_rect_with_circle(X0, -wx, Z0, Z1, yf, -xo, zc, ro), CU, normal=[0, -1, 0])
    S.add(keyhole_rect_with_circle(wx, X1, Z0, Z1, yf, xo, zc, ro), CU, normal=[0, -1, 0])
    S.add([[-wx, yf, Z0], [wx, yf, Z0], [wx, yf, zc - wz], [-wx, yf, zc - wz]], CU, normal=[0, -1, 0])
    S.add([[-wx, yf, zc + wz], [wx, yf, zc + wz], [wx, yf, Z1], [-wx, yf, Z1]], CU, normal=[0, -1, 0])
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, yf - 0.02, zc - gh], [xb, yf - 0.02, zc - gh], [xb, yf - 0.02, zc + gh], [xa, yf - 0.02, zc + gh]], CU_DARK * 0.5, flat=True)
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["top", "right"])
    out.append((S, 4))
    # 5) optical beam along y: behind the block (occluded by it), inside the pocket
    #    (seen through the front opening) and in front of the block
    beam = np.array([0.85, 0.05, 0.15])
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.35, Y1 + 0.05, 32, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.9)
    out.append((S, 1.5))
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.35, Y0, Y1, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.9)
    out.append((S, 3.5))
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.35, -22, Y0 - 0.05, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.9)
    out.append((S, 5))
    return out, dict(zc=zc, xo=xo, ro=ro, wx=wx, wz=wz, X0=X0, X1=X1, Y0=Y0, Y1=Y1, Z1=Z1)


def sc_scene():
    """Planar superconducting resonator on a thin CaWO4 chip (proposed, not built)."""
    S = Scene()
    S.box(-3, 3, -3, 3, 0, 0.5, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.4, alpha=0.75, faces=["top", "front", "right", "left", "back"])
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
    return S


def draw_spins_2d(fig, ax, pts3d, colour=ORANGE, size=0.008):
    for p in pts3d:
        d = fig.transFigure.inverted().transform(p2d(ax, *p))
        fig.add_artist(FancyArrowPatch((d[0], d[1] - size), (d[0], d[1] + size), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=4, lw=0.5, color=colour))


def marker(fig, xy, num, colour="k", r=0.011, fs=6.0):
    """Circled number placed at figure coordinates xy."""
    fig.add_artist(matplotlib.patches.Circle(xy, r, transform=fig.transFigure, fc="white", ec=colour, lw=0.7, zorder=30))
    fig.text(xy[0], xy[1], str(num), fontsize=fs, ha="center", va="center", color=colour, zorder=31, fontweight="bold")


def marker3d(fig, ax, xyz, num, offset=(0.0, 0.0), colour="k"):
    """Circled number next to a 3-D point, joined by a short leader if offset."""
    p = fig.transFigure.inverted().transform(p2d(ax, *xyz))
    q = (p[0] + offset[0], p[1] + offset[1])
    if abs(offset[0]) + abs(offset[1]) > 0.012:
        fig.add_artist(FancyArrowPatch(q, p, transform=fig.transFigure, arrowstyle="-", lw=0.6, color=colour, shrinkA=4, shrinkB=0, zorder=29))
    marker(fig, q, num, colour=colour)
    return q


def text3d(fig, ax, xyz, txt, colour="k", fontsize=6.0, ha="center", va="center", rotation=0, **kw):
    """Text placed directly on a 3-D point (for annotating a layer in place)."""
    p = fig.transFigure.inverted().transform(p2d(ax, *xyz))
    fig.text(p[0], p[1], txt, fontsize=fontsize, ha=ha, va=va, color=colour, rotation=rotation, zorder=32, **kw)


# ---------------------------------------------------------------------------
def make():
    fig = plt.figure(figsize=(7.0, 3.9))

    # ---------------- (a) loop-gap resonator ----------------
    ax = fig.add_axes([-0.03, 0.17, 0.65, 0.88], projection="3d")
    ax.computed_zorder = False
    scenes, G = loop_gap_scene()
    for S, zo in scenes:
        S.draw(ax, zorder=zo)
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-112)
    ax.set_xlim(-27, 27)
    ax.set_ylim(-19, 20)
    ax.set_zlim(-3, 22)
    ax.set_box_aspect((54, 39, 25))
    ax.set_axis_off()
    fig.canvas.draw()
    zc, xo, ro, wx, wz = G["zc"], G["xo"], G["ro"], G["wx"], G["wz"]
    Y0, Y1, X0, X1, Z1 = G["Y0"], G["Y1"], G["X0"], G["X1"], G["Z1"]

    fig.text(0.008, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
    # in-layer annotations
    text3d(fig, ax, (0, Y1, Z1 + 1.5), "loop-gap microwave resonator", colour="k", fontsize=6.4, va="bottom", fontweight="bold")
    text3d(fig, ax, (0, 0.5 * (Y0 + Y1), Z1), r"$\kappa/2\pi = 660$ kHz, $V_{\rm m} = 275$ mm$^3$", colour="w", fontsize=5.6, va="center")
    text3d(fig, ax, (0, -11, 0), "mixing-chamber plate of a dilution refrigerator, base temperature < 30 mK", colour="k", fontsize=6.0, va="center")
    text3d(fig, ax, (0, Y0, zc - wz - 0.9), "central loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (-xo, Y0, zc - ro - 0.8), "outer loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (xo, Y0, zc - ro - 0.8), "outer loop", colour="w", fontsize=5.4, va="top")
    text3d(fig, ax, (-(xo - ro + wx) / 2, Y0, zc - 1.4), "gap", colour="w", fontsize=5.2, va="top")
    text3d(fig, ax, ((xo - ro + wx) / 2, Y0, zc - 1.4), "gap", colour="w", fontsize=5.2, va="top")
    # numbered markers close to the components
    marker3d(fig, ax, (0, Y0 - 0.5, zc - 0.5), 1, offset=(-0.045, -0.065))          # crystal
    marker3d(fig, ax, (X0 + 3, Y0, Z1 - 2.5), 2, offset=(0.0, 0.0))                   # resonator
    marker3d(fig, ax, (0, -20, zc), 3, offset=(0.0, -0.045), colour=RED)              # beam
    marker3d(fig, ax, (xo, Y0 - 1, zc), 4, offset=(0.04, -0.05), colour=BLUE)         # microwave port
    marker3d(fig, ax, (-xo, Y0 - 1, zc), 4, offset=(-0.045, -0.03), colour=BLUE)
    marker3d(fig, ax, (23, -13, 0), 5, offset=(0.0, 0.0))                             # plate
    # microwave I/O: wavy arrows into the outer loops (as in the source figure)
    for xc, sgn in [(-xo, -1), (xo, 1)]:
        p = fig.transFigure.inverted().transform(p2d(ax, xc, Y0 - 0.5, zc))
        wavy(fig, (p[0] + sgn * 0.06, p[1] - 0.075), (p[0], p[1] - 0.008), BLUE, n=4, amp=0.004, lw=1.0)
    # process glyphs at the crystal
    c0 = fig.transFigure.inverted().transform(p2d(ax, 0, Y0 - 0.5, zc + 1.0))
    fig.add_artist(FancyArrowPatch((c0[0] - 0.03, c0[1] - 0.006), (c0[0] + 0.03, c0[1] + 0.006), transform=fig.transFigure,
                                   arrowstyle="<|-|>", mutation_scale=7, lw=1.1, color=ORANGE, zorder=20))
    wavy(fig, (c0[0] + 0.02, c0[1] + 0.03), (c0[0] + 0.075, c0[1] + 0.085), BLUE, n=4, amp=0.004, lw=0.9)
    wavy(fig, (c0[0] - 0.085, c0[1] + 0.08), (c0[0] - 0.03, c0[1] + 0.03), PINK, n=4, amp=0.004, lw=0.9)
    fig.add_artist(FancyArrowPatch((c0[0] + 0.012, c0[1] - 0.02), (c0[0] + 0.012, c0[1] - 0.055), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, lw=0.9, color=GREEN, zorder=20))

    # ---------------- key (right column, top) ----------------
    lx, ly, lw_, lh = 0.605, 0.455, 0.39, 0.535
    fig.add_artist(FancyBboxPatch((lx, ly), lw_, lh, boxstyle="round,pad=0.005", transform=fig.transFigure, fc="#f6f6f6", ec="0.5", lw=0.6))
    y = ly + lh - 0.02
    fig.text(lx + 0.01, y, "Key to (a), loop-gap device as built [Fukumori $et\\ al.$]:", fontsize=6.3, va="center", fontweight="bold")
    y -= 0.006
    items = [
        (1, "k", "$^{171}$Yb$^{3+}$:CaWO$_4$ crystal, 4.4 $\\times$ 4.6 $\\times$ 5 mm, 4.96 ppm $^{171}$Yb,\nglued in the central loop, $c$ axis (polished faces) along the beam"),
        (2, "k", "loop-gap resonator: central loop, two outer loops, narrow gaps;\n$\\kappa/2\\pi = 660$ kHz, $V_{\\rm m} = 275$ mm$^3$, $g/2\\pi = 15$ mHz"),
        (3, RED, "973 nm optical I/O (fibre collimators): pumping into $|\\!\\downarrow\\rangle$, absorption\nreadout of $|\\!\\uparrow\\rangle$, $|\\!\\downarrow\\rangle$; the beam covers about 1% of the crystal face"),
        (4, BLUE, "microwave I/O antennas at the outer loops: $\\pi/2$, $\\pi$ pulses;\ntransmission $S_{21}$ calibrates $N_0$"),
        (5, "k", "mixing-chamber plate, base < 30 mK; spin temperature 80 mK"),
    ]
    y -= 0.03
    for num, col, txt in items:
        marker(fig, (lx + 0.022, y), num, colour=col)
        fig.text(lx + 0.04, y, txt, fontsize=5.5, va="center", linespacing=1.15)
        y -= 0.056 if "\n" in txt else 0.036
    y += 0.006
    fig.text(lx + 0.01, y, "Spin and operating parameters:", fontsize=6.3, va="center", fontweight="bold")
    y -= 0.026
    for txt in ["$\\omega_s/2\\pi = 3.084$ GHz (zero-field clock transition), $\\gamma_{\\rm inh}/2\\pi < 5$ kHz, $T_2 > 150$ ms",
                "$\\Delta/2\\pi = 22$ MHz, $2\\Delta/\\kappa = 67$, $N_0 \\leq 7\\times10^{14}$, $n_{\\rm th} = 0.19$ at 80 mK",
                "model uses $\\gamma_{\\rm inh}/2\\pi = 5$ kHz (Voigt, 30% Lorentzian) and $T_2 = 150$ ms"]:
        fig.text(lx + 0.012, y, txt, fontsize=5.5, va="center")
        y -= 0.026
    y += 0.004
    fig.text(lx + 0.01, y, "Processes kept in the model (glyphs at the crystal):", fontsize=6.3, va="center", fontweight="bold")
    y -= 0.026
    procs = [
        (ORANGE, "<|-|>", "one-axis twisting $\\chi\\,\\hat J_+\\hat J_-$ (cavity-mediated exchange)"),
        (BLUE, "wave", "collective emission through the detuned cavity, $\\Gamma_{\\rm SR}$"),
        (PINK, "wave", "thermal cavity photons absorbed collectively, $\\Gamma_{\\rm SR}\\,n_{\\rm th}$"),
        (GREEN, "-|>", "single-spin dephasing, $1/T_2$"),
    ]
    for col, kind, txt in procs:
        if kind == "wave":
            wavy(fig, (lx + 0.012, y), (lx + 0.034, y), col, n=3, amp=0.0035, lw=0.9)
        else:
            fig.add_artist(FancyArrowPatch((lx + 0.012, y), (lx + 0.034, y), transform=fig.transFigure, arrowstyle=kind, mutation_scale=6, lw=0.9, color=col))
        fig.text(lx + 0.04, y, txt, fontsize=5.5, va="center")
        y -= 0.026

    # ---------------- level-structure inset (bottom left) ----------------
    ix = fig.add_axes([0.02, 0.01, 0.27, 0.20])
    ix.set_xlim(0, 10)
    ix.set_ylim(0, 10)
    ix.axis("off")
    ix.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7, boxstyle="round,pad=0.1", fc="white", ec="0.5", lw=0.6))
    ix.text(5, 9.1, "levels of one $^{171}$Yb$^{3+}$ ion used here", ha="center", va="center", fontsize=6.0)
    ix.plot([1.8, 4.4], [1.8, 1.8], color="k", lw=1.2)
    ix.plot([1.8, 4.4], [3.6, 3.6], color="k", lw=1.2)
    ix.text(4.6, 1.8, r"$|\!\downarrow\rangle$", va="center", fontsize=7)
    ix.text(4.6, 3.6, r"$|\!\uparrow\rangle$", va="center", fontsize=7)
    ix.annotate("", xy=(3.6, 3.55), xytext=(3.6, 1.85), arrowprops=dict(arrowstyle="<->", lw=0.6, color=BLUE))
    ix.text(1.6, 2.7, "3.084 GHz\nclock\ntransition", fontsize=5.2, va="center", ha="right", color=BLUE)
    ix.plot([1.8, 4.4], [7.4, 7.4], color="k", lw=1.2)
    ix.text(4.6, 7.4, r"$|e\rangle$", va="center", fontsize=7)
    ix.annotate("", xy=(2.4, 7.35), xytext=(2.4, 1.85), arrowprops=dict(arrowstyle="<->", lw=0.7, color=RED))
    ix.annotate("", xy=(3.0, 7.35), xytext=(3.0, 3.65), arrowprops=dict(arrowstyle="<->", lw=0.7, color=RED))
    ix.text(1.6, 5.8, "973 nm\nA, E", fontsize=5.2, color=RED, ha="right", va="center")
    ix.text(7.4, 5.3, "A, E: optical pumping\nand absorption readout\nof the two populations\n\n" + r"$\pi/2$, $\pi$ microwave pulses" + "\nthrough the resonator", fontsize=5.1, ha="center", va="center", linespacing=1.15)

    # ---------------- (b) superconducting resonator (proposed) ----------------
    bx = fig.add_axes([0.59, 0.14, 0.36, 0.31], projection="3d")
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
    fig.text(0.607, 0.445, "(b)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.64, 0.443, "planar superconducting resonator on the same crystal:\nthe design-map case (proposed, not built)", fontsize=6.0, va="top", linespacing=1.15)
    text3d(fig, bx, (-2.2, 0.0, 0.55), "pad", colour="w", fontsize=5.2)
    text3d(fig, bx, (2.2, 0.0, 0.55), "pad", colour="w", fontsize=5.2)
    text3d(fig, bx, (0, 3.2, 0.9), "CaWO$_4$ chip, $N \\approx 10^{10}$ spins", colour=CRYSTAL_EDGE, fontsize=5.6, ha="center", va="bottom")
    text3d(fig, bx, (0, -3.4, -0.2), "Nb meander inductor; optical readout spot (red)\nover the near-uniform field region (coupling spread $D$)", colour="k", fontsize=5.4, va="top", ha="center")
    fig.text(0.995, 0.31, r"$\kappa/2\pi = 3$ to 300 kHz" + "\n" + r"$g\sqrt{N}/2\pi = 0.1$ to 5 MHz" + "\n20 mK, $T_2 = 150$ ms", fontsize=5.8, ha="right", va="center", linespacing=1.2)

    # ---------------- (c) pulse sequences ----------------
    cx = fig.add_axes([0.655, 0.012, 0.335, 0.12])
    cx.set_xlim(-2.6, 10)
    cx.set_ylim(0, 4.4)
    cx.axis("off")
    fig.text(0.607, 0.135, "(c)", fontsize=9, fontweight="bold", va="top")
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
