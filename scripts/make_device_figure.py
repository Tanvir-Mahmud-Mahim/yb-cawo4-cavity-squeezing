"""Figure 1: three-dimensional device schematics.

Panel (a) follows the system overview of Fukumori et al. (arXiv:2604.26909,
Fig. 1a): a metal block with a central loop (rectangular pocket) that holds the
crystal, two circular outer loops on either side connected to it by narrow gaps,
the optical beam through the crystal along the loop axis, and microwave
antennas at the outer loops.  Every number written on the figure is taken from
that paper (crystal 4.4 x 4.6 x 5 mm, 4.96 ppm 171Yb, 973 nm, kappa/2pi = 660 kHz,
g/2pi = 15 mHz, V_m = 275 mm^3, base temperature < 30 mK, spin temperature 80 mK).
Exact proportions of the block are schematic.  Panel (b) is a magnified crystal
showing the four processes kept in the model.  Panel (c) is the planar
superconducting resonator considered in the design map (a proposal, not a
fabricated device).
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
        if edge == "face":
            self.ec.append((*col, alpha))
            lw = max(lw, 0.6)
        else:
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
def band_with_circle(x0, x1, z0, z1, y, cx, cz, r, n=24):
    """Front-face band (plane y = const) with a circular hole, as four polygons
    (top, bottom, left C-piece, right C-piece) so that no slit is needed."""
    tl = np.linspace(1.5 * np.pi, 0.5 * np.pi, n + 1)        # left half circle, through 180 deg
    tr = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n + 1)       # right half circle, through 0 deg
    top = [[x0, y, cz + r], [x1, y, cz + r], [x1, y, z1], [x0, y, z1]]
    bot = [[x0, y, z0], [x1, y, z0], [x1, y, cz - r], [x0, y, cz - r]]
    left = [[x0, y, cz - r], [cx, y, cz - r]] + [[cx + r * np.cos(t), y, cz + r * np.sin(t)] for t in tl] + [[x0, y, cz + r]]
    right = [[cx, y, cz - r], [x1, y, cz - r], [x1, y, cz + r], [cx, y, cz + r]] + [[cx + r * np.cos(t), y, cz + r * np.sin(t)] for t in tr[::-1]]
    return [top, bot, left, right]


def loop_gap_scene():
    """Loop-gap resonator after Fukumori et al., Fig. 1a: an opaque metal block with a
    central rectangular loop (crystal pocket, open at the front), two circular outer
    loops and thin gaps, all running through the block along y.  Thin edge lines
    give a crisp drawing.  Returns [(scene, zorder), ...]."""
    out = []
    X0, X1, Y0, Y1, Z0, Z1 = -17, 17, 0, 12, 0, 20
    zc = 10.0
    wx, wz = 6.0, 7.0
    ro, xo = 3.0, 12.0
    gh = 0.35
    E = tuple(CU * 0.45)          # edge colour of the block
    EP = tuple(PLATE * 0.55)
    # 1) plate
    S = Scene()
    S.box(-26, 26, -16, 18, -2.0, 0.0, PLATE, faces=["top", "front", "left", "right"], edge=EP, lw=0.35)
    out.append((S, 1))
    # 2) far parts of the block: back, bottom, left end, inner walls
    S = Scene()
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["back", "bottom", "left"])
    for xc in (-xo, xo):
        cylinder_wall(S, "y", (xc, zc), ro, Y0, Y1, 0, 2 * np.pi, CU_DARK, n=40, inner=True)
    for poly, nrm in [
        ([[-wx, Y0, zc - wz], [wx, Y0, zc - wz], [wx, Y1, zc - wz], [-wx, Y1, zc - wz]], [0, 0, 1]),
        ([[-wx, Y0, zc + wz], [wx, Y0, zc + wz], [wx, Y1, zc + wz], [-wx, Y1, zc + wz]], [0, 0, -1]),
        ([[-wx, Y0, zc - wz], [-wx, Y1, zc - wz], [-wx, Y1, zc + wz], [-wx, Y0, zc + wz]], [1, 0, 0]),
        ([[wx, Y0, zc - wz], [wx, Y1, zc - wz], [wx, Y1, zc + wz], [wx, Y0, zc + wz]], [-1, 0, 0]),
        ([[-wx, Y1, zc - wz], [wx, Y1, zc - wz], [wx, Y1, zc + wz], [-wx, Y1, zc + wz]], [0, -1, 0]),   # back wall of the pocket
    ]:
        S.add(poly, CU_DARK, normal=nrm)
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, Y0, zc - gh], [xb, Y0, zc - gh], [xb, Y1, zc - gh], [xa, Y1, zc - gh]], CU_DARK * 0.7, normal=[0, 0, 1])
        S.add([[xa, Y0, zc + gh], [xb, Y0, zc + gh], [xb, Y1, zc + gh], [xa, Y1, zc + gh]], CU_DARK * 0.7, normal=[0, 0, -1])
    out.append((S, 2))
    # 3) crystal glued at the centre of the pocket (4.4 x 4.6 x 5 mm, c axis along y)
    S = Scene()
    cy = 0.5 * (Y0 + Y1)
    S.box(-2.2, 2.2, cy - 2.5, cy + 2.5, zc - 2.3, zc + 2.3, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.5, alpha=0.55)
    out.append((S, 3))
    # 4) beam inside the pocket
    beam = np.array([0.85, 0.05, 0.15])
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.4, Y0, Y1, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.5)
    out.append((S, 3.5))
    # 5) front face with the openings, top and right faces
    S = Scene()
    yf = Y0
    for piece in band_with_circle(X0, -wx, Z0, Z1, yf, -xo, zc, ro) + band_with_circle(wx, X1, Z0, Z1, yf, xo, zc, ro):
        S.add(piece, CU, normal=[0, -1, 0], edge="face")
    S.add([[-wx, yf, Z0], [wx, yf, Z0], [wx, yf, zc - wz], [-wx, yf, zc - wz]], CU, normal=[0, -1, 0], edge="face")
    S.add([[-wx, yf, zc + wz], [wx, yf, zc + wz], [wx, yf, Z1], [-wx, yf, Z1]], CU, normal=[0, -1, 0], edge="face")
    for xa, xb in [(-xo + ro, -wx), (wx, xo - ro)]:
        S.add([[xa, yf - 0.02, zc - gh], [xb, yf - 0.02, zc - gh], [xb, yf - 0.02, zc + gh], [xa, yf - 0.02, zc + gh]], CU_DARK * 0.5, flat=True)
    S.box(X0, X1, Y0, Y1, Z0, Z1, CU, faces=["top", "right"])
    out.append((S, 4))
    # 6) beam in front of the block
    S = Scene()
    cylinder_wall(S, "y", (0, zc), 0.4, -19, Y0 - 0.05, 0, 2 * np.pi, beam, n=14, inner=False, alpha=0.5)
    out.append((S, 5))
    # 7) visible edge lines (block outline, hole rims, pocket rims)
    th = np.linspace(0, 2 * np.pi, 80)
    edges = [
        [[X0, Y0, Z0], [X1, Y0, Z0], [X1, Y0, Z1], [X0, Y0, Z1], [X0, Y0, Z0]],          # front outline
        [[X0, Y0, Z1], [X0, Y1, Z1], [X1, Y1, Z1], [X1, Y0, Z1]],                        # top outline
        [[X1, Y1, Z1], [X1, Y1, Z0], [X1, Y0, Z0]],                                      # right face
        [[X0, Y0, Z0], [X0, Y1, Z0], [X0, Y1, Z1]],                                      # left face (visible part)
        [[-wx, Y0, zc - wz], [wx, Y0, zc - wz], [wx, Y0, zc + wz], [-wx, Y0, zc + wz], [-wx, Y0, zc - wz]],   # pocket front rim
        [[-wx, Y1, zc - wz], [wx, Y1, zc - wz]],                                                          # pocket back floor edge
        [[-wx, Y0, zc - wz], [-wx, Y1, zc - wz]], [[wx, Y0, zc - wz], [wx, Y1, zc - wz]],                   # floor edges
    ]
    for xc in (-xo, xo):
        edges.append([[xc + ro * np.cos(t), Y0, zc + ro * np.sin(t)] for t in th])
    return out, dict(zc=zc, xo=xo, ro=ro, wx=wx, wz=wz, X0=X0, X1=X1, Y0=Y0, Y1=Y1, Z1=Z1, cy=cy, edges=edges, edge_colour=E)


def crystal_zoom_scene():
    """Magnified crystal for the process panel."""
    S = Scene()
    S.box(-2.2, 2.2, -2.5, 2.5, -2.3, 2.3, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.6, alpha=0.35)
    return S


def sc_scene():
    """Planar superconducting resonator on a thin CaWO4 chip (proposed, not built)."""
    out = []
    S = Scene()
    S.box(-3, 3, -3, 3, 0, 0.5, CRYSTAL, edge=CRYSTAL_EDGE, lw=0.4, alpha=0.35, faces=["bottom", "back", "left"])
    out.append((S, 1))
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
    w = 0.16
    ys = np.linspace(-1.8, 1.8, 9)
    for k, yy in enumerate(ys):
        S.box(-1.7, 1.7, yy - w / 2, yy + w / 2, 0.5, t, NB, faces=["top", "front"])
        if k < len(ys) - 1:
            xa = 1.7 - w if k % 2 == 0 else -1.7
            S.box(xa, xa + w, yy - w / 2, ys[k + 1] + w / 2, 0.5, t, NB)
    out.append((S, 4))
    S = Scene()
    cylinder_wall(S, "z", (0, 0), 0.45, -0.6, 2.4, 0, 2 * np.pi, np.array([0.85, 0.05, 0.15]), n=16, inner=False, alpha=0.35)
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
    return p


def setup3d(ax, xlim, ylim, zlim, aspect, zoom, elev, azim):
    ax.computed_zorder = False
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(aspect, zoom=zoom)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
def make():
    fig = plt.figure(figsize=(7.0, 3.8))

    # ================= (a) loop-gap resonator experiment =================
    ax = fig.add_axes([-0.02, 0.30, 0.56, 0.70], projection="3d")
    scenes, G = loop_gap_scene()
    for S, zo in scenes:
        S.draw(ax, zorder=zo)
    setup3d(ax, (-27, 27), (-19, 20), (-3, 23), (54, 39, 26), 1.55, 18, -112)
    for e in G["edges"]:
        e = np.asarray(e, float)
        ln, = ax.plot(e[:, 0], e[:, 1], e[:, 2], color=G["edge_colour"], lw=0.45, solid_capstyle="round")
        ln.set_zorder(6)
    fig.canvas.draw()
    zc, xo, ro, wx, wz, cy = G["zc"], G["xo"], G["ro"], G["wx"], G["wz"], G["cy"]
    Y0, Y1, X0, X1, Z1 = G["Y0"], G["Y1"], G["X0"], G["X1"], G["Z1"]
    rng = np.random.default_rng(5)
    spins = [tuple(v) for v in rng.uniform([-1.7, cy - 2.0, zc - 1.8], [1.7, cy + 2.0, zc + 1.8], size=(10, 3))]
    draw_spins_2d(fig, ax, spins, colour=CRYSTAL_EDGE, size=0.006, lw=0.5, ms=3.5)

    q0, q1 = p2d(ax, 0, Y0, zc), p2d(ax, 10, Y0, zc)
    ang_x = float(np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0])))
    fig.text(0.008, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.04, 0.985, "the demonstrated experiment: crystal in a loop-gap microwave resonator", fontsize=6.6, va="top")
    # labels written on the parts themselves
    text3d(fig, ax, (0, 0.5 * (Y0 + Y1), Z1), r"$\kappa/2\pi = 660$ kHz", colour="w", fontsize=5.8, va="center", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (0, -11, 0), "mixing-chamber plate, below 30 mK", colour="k", fontsize=5.8, va="center", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (0, Y0, zc - wz - 1.0), "central loop", colour="w", fontsize=5.4, va="top", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (-xo - 1.0, Y0, zc + ro + 0.9), "outer loop", colour="w", fontsize=5.4, va="bottom", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (xo + 1.0, Y0, zc + ro + 0.9), "outer loop", colour="w", fontsize=5.4, va="bottom", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (-wx - 1.5, Y0, zc - 1.0), "gap", colour="w", fontsize=5.0, va="top", ha="center", rotation=ang_x, rotation_mode="anchor")
    text3d(fig, ax, (wx + 1.5, Y0, zc - 1.0), "gap", colour="w", fontsize=5.0, va="top", ha="center", rotation=ang_x, rotation_mode="anchor")
    # numbered markers placed on the parts (no long leaders)
    marker3d(fig, ax, (4.6, Y0 + 2.0, zc + 4.6), 1)
    marker3d(fig, ax, (X0 + 3.5, Y0, Z1 - 3.0), 2)
    marker3d(fig, ax, (0, -16.5, zc), 3, colour=RED)
    marker3d(fig, ax, (-xo, Y0 - 0.2, zc - ro - 3.2), 4, colour=BLUE)
    marker3d(fig, ax, (xo, Y0 - 0.2, zc - ro - 3.2), 4, colour=BLUE)
    marker3d(fig, ax, (17, -13, 0), 5)
    # microwave signals entering the outer loops, as in the source figure
    for xc, sgn in [(-xo, -1), (xo, 1)]:
        p = fig.transFigure.inverted().transform(p2d(ax, xc, Y0 - 0.5, zc))
        wavy(fig, (p[0] + sgn * 0.075, p[1] - 0.06), (p[0], p[1] - 0.006), BLUE, n=4, amp=0.004, lw=1.0)
    # dashed frame around the crystal, referring to the magnified view (b)
    # single dashed rectangle in the plane of the pocket opening (in front of the crystal),
    # drawn in perspective, referring to (b)
    mx, mz = 1.3, 1.3
    yf = cy - 2.5 - 0.3
    rect = np.array([[-2.2 - mx, yf, zc - 2.3 - mz], [2.2 + mx, yf, zc - 2.3 - mz],
                     [2.2 + mx, yf, zc + 2.3 + mz], [-2.2 - mx, yf, zc + 2.3 + mz], [-2.2 - mx, yf, zc - 2.3 - mz]])
    ln, = ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color="0.92", lw=0.6, ls=(0, (3, 2)))
    ln.set_zorder(7)
    pf = fig.transFigure.inverted().transform(p2d(ax, 2.2 + mx, yf, zc - 2.3 - mz - 0.3))
    fig.text(pf[0], pf[1], "see (b)", fontsize=5.4, color="0.92", ha="right", va="top", zorder=33, rotation=ang_x, rotation_mode="anchor")

    # ================= (b) magnified crystal with the four processes =================
    bx = fig.add_axes([0.545, 0.555, 0.24, 0.38], projection="3d")
    crystal_zoom_scene().draw(bx, zorder=2)
    setup3d(bx, (-3.2, 3.2), (-3.2, 3.2), (-3.2, 3.2), (1, 1, 1), 1.05, 18, -112)
    fig.canvas.draw()
    rng = np.random.default_rng(7)
    spins = [tuple(v) for v in rng.uniform([-1.7, -2.0, -1.8], [1.7, 2.0, 1.8], size=(12, 3))]
    draw_spins_2d(fig, bx, spins, colour=CRYSTAL_EDGE, size=0.009, lw=0.6, ms=4.5)
    fig.text(0.565, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.595, 0.985, "what the model follows inside the crystal", fontsize=6.6, va="top")
    c = fig.transFigure.inverted().transform(p2d(bx, 0, 0, 0))
    cr = fig.transFigure.inverted().transform(p2d(bx, 2.2, -2.5, 0))     # right edge of the cube, mid height
    ct = fig.transFigure.inverted().transform(p2d(bx, 0, 0, 2.3))        # top
    cb = fig.transFigure.inverted().transform(p2d(bx, 0, 0, -2.3))       # bottom
    lx0 = cr[0] + 0.04                                                    # label column to the right of the cube
    # twisting: two highlighted spins joined by a double arrow
    for p in [(c[0] - 0.035, c[1] - 0.008), (c[0] + 0.035, c[1] + 0.008)]:
        fig.add_artist(FancyArrowPatch((p[0], p[1] - 0.011), (p[0], p[1] + 0.011), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=6, lw=1.0, color=ORANGE, zorder=27))
    fig.add_artist(FancyArrowPatch((c[0] - 0.029, c[1] - 0.007), (c[0] + 0.029, c[1] + 0.007), transform=fig.transFigure,
                                   arrowstyle="<|-|>", mutation_scale=8, lw=1.3, color=ORANGE, zorder=26))
    fig.add_artist(matplotlib.lines.Line2D([c[0] + 0.045, lx0 - 0.005], [c[1] + 0.008, c[1] + 0.008], transform=fig.transFigure, color=ORANGE, lw=0.5, zorder=26))
    fig.text(lx0, c[1] + 0.008, "spins twist each other through\nthe resonator (strength $\\chi$)", fontsize=5.6, color=ORANGE, ha="left", va="center", linespacing=1.15)
    # collective emission: wave leaving the top right corner
    wavy(fig, (c[0] + 0.02, c[1] + 0.04), (cr[0] + 0.02, ct[1] + 0.005), BLUE, n=5, amp=0.005, lw=1.1)
    fig.text(lx0, ct[1] + 0.012, "collective emission into the\nresonator (rate $\\Gamma_{\\rm SR}$)", fontsize=5.6, color=BLUE, ha="left", va="center", linespacing=1.15)
    # thermal photons: wave entering the top left corner
    wavy(fig, (c[0] - 0.085, ct[1] + 0.03), (c[0] - 0.03, c[1] + 0.045), PINK, n=5, amp=0.005, lw=1.1)
    fig.text(c[0] - 0.09, ct[1] + 0.045, "thermal photons from the\nresonator (rate $\\Gamma_{\\rm SR}n_{\\rm th}$)", fontsize=5.6, color=PINK, ha="left", va="bottom", linespacing=1.15)
    # dephasing: one spin at the bottom right losing its phase
    fig.add_artist(FancyArrowPatch((c[0] + 0.02, c[1] - 0.035), (c[0] + 0.02, c[1] - 0.085), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=1.1, color=GREEN, zorder=26))
    fig.add_artist(matplotlib.lines.Line2D([c[0] + 0.028, lx0 - 0.005], [c[1] - 0.07, c[1] - 0.07], transform=fig.transFigure, color=GREEN, lw=0.5, zorder=26))
    fig.text(lx0, c[1] - 0.07, "each spin loses its phase\non its own (rate $1/T_2$)", fontsize=5.6, color=GREEN, ha="left", va="center", linespacing=1.15)
    fig.text(c[0], cb[1] - 0.04, "spins: the 3.084 GHz clock transition of $^{171}$Yb$^{3+}$", fontsize=5.6, color=CRYSTAL_EDGE, ha="center", va="top")

    # ================= (c) proposed planar superconducting resonator =================
    cx3 = fig.add_axes([0.55, 0.02, 0.46, 0.50], projection="3d")
    for S, zo in sc_scene():
        S.draw(cx3, zorder=zo)
    setup3d(cx3, (-3.4, 3.4), (-3.4, 3.4), (-1.2, 2.8), (6.8, 6.8, 4.0), 1.45, 26, -52)
    fig.canvas.draw()
    rng = np.random.default_rng(3)
    spins = [tuple(v) for v in rng.uniform([-1.4, -1.6, 0.1], [1.4, 1.6, 0.42], size=(16, 3))]
    draw_spins_2d(fig, cx3, spins, colour=ORANGE, size=0.007, lw=0.5, ms=4)
    fig.text(0.565, 0.50, "(c)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.595, 0.50, "proposed planar superconducting resonator on the same crystal (not built)", fontsize=6.6, va="top")
    marker3d(fig, cx3, (-2.95, -2.9, 0.3), 6, offset=(-0.028, -0.015))          # chip
    marker3d(fig, cx3, (1.7, -1.9, 0.25), 7, offset=(0.035, -0.03))              # read-out volume
    marker3d(fig, cx3, (1.2, 1.9, 0.52), 8, offset=(0.045, 0.03))                # meander inductor
    marker3d(fig, cx3, (-2.2, 0.0, 0.52), 9)                                     # left pad
    marker3d(fig, cx3, (2.2, 0.0, 0.52), 9)                                      # right pad
    marker3d(fig, cx3, (0.45, 0, 2.2), 10, offset=(0.04, 0.0), colour=RED)       # optical read-out beam

    # ================= key =================
    kx, ky = 0.012, 0.275
    fig.text(kx, ky, "Key", fontsize=6.4, fontweight="bold", va="center")
    items = [
        (1, "k", "$^{171}$Yb$^{3+}$:CaWO$_4$ crystal, 4.4 $\\times$ 4.6 $\\times$ 5 mm, 4.96 ppm $^{171}$Yb, glued in the central loop"),
        (2, "k", "metal loop-gap resonator: central loop, two outer loops, narrow gaps ($g/2\\pi = 15$ mHz, $V_{\\rm m} = 275$ mm$^3$)"),
        (3, RED, "973 nm light through the crystal: optical pumping into $|\\!\\downarrow\\rangle$ and absorption readout of the populations"),
        (4, BLUE, "microwave antennas at the outer loops: spin control pulses and transmission measurement"),
        (5, "k", "mixing-chamber plate of the dilution refrigerator (spin temperature 80 mK, $n_{\\rm th} = 0.19$)"),
        (6, "k", "CaWO$_4$ chip carrying the resonator"),
        (7, "k", "read-out volume under the inductor, where the microwave field is nearly uniform (coupling spread $D$)"),
        (8, "k", "niobium meander inductor (the part of the resonator that couples to the spins)"),
        (9, "k", "niobium capacitor pads"),
        (10, RED, "optical read-out beam through the read-out volume"),
    ]
    y = ky - 0.028
    for num, col, txt in items:
        marker(fig, (kx + 0.012, y), num, colour=col, r=0.0095, fs=5.2)
        fig.text(kx + 0.028, y, txt, fontsize=5.3, va="center")
        y -= 0.0265

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_device.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("figure fig_device")


if __name__ == "__main__":
    make()
