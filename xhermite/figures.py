"""
Figures 1-5 of the article, regenerated from the original Mathematica data.

Called by ``python -m xhermite.article`` (and hence by ``make results`` /
``make full`` in the article directory); writes vector PDFs to
``<outdir>/figuras/``. Data sources (original notebooks, kept in
``reference/mathematica`` or ``Mathematica_referencia``):

* fig1_basis_*.pdf -- basis H_q^3(t) for q = -0.8, 0.8, 1, 2.5; plot ranges from
             ``splines_q-bases.nb`` (cells ``l = ...``, ``rngX = ...``).
* fig2_curve_*.pdf -- parametric x-Hermite spline (x_q(t), y_q(t)), 3 segments,
             ``splines_q-bases.nb``: ptx = {{0.8,2.5},{2.3,0.5},{4,0.9},{1,-1}},
             pty = {{1,1},{3.6,-1},{2,7},{2,-1}} (pairs = position, tangent),
             tangents used directly in the local parameter t.
* fig3_weierstrass.pdf -- W_2 .. W_5 on [0, 1] (K = 250).
* fig4_constants_*.pdf -- W_3 sampled at 13 uniform nodes x = 0..12 (argument x/12),
             zero tangents, q = 2.9 and q = -0.6 (``1DTensorBSplines.nb``,
             level-amplifying / level-attenuating examples).
* fig5a_global.pdf, fig5b_per_segment.pdf -- W_3, N_P = 7, zero tangents; (a) q = 2.6, (b) one q per segment
             {2.9, 1.4, -0.2, 2.8, 2.1, -0.5}.

Every curve is evaluated with the factorized form of Theorem 3(ii)
(``xhermite_segment``), which equals the notebook's R M_q H_q^3(t).
"""
from __future__ import annotations

import os

import numpy as np

from .mathematica_reference import xhermite_segment, weierstrass_rational

# Visual style follows the original Mathematica figures: solid lines only,
# Mathematica's default plot colours (indigo, magenta, olive, green), axes
# through the origin, curves identified by floating labels (Fig. 1, Fig. 2
# points) or by a legend (Figs. 3-5). Each panel is drawn at its printed
# size (width in the LaTeX source), so fonts and line widths are uniform.
MMA = ("#3d3d99", "#99406b", "#998040", "#3d9959")   # Mathematica 8 default
C_W, C_CLASS_F4, C_Q_F4 = "#e62e2e", "#2f55d4", "#99406b"
C_CLASS_F5, C_Q_F5 = "black", "#2f55d4"
FONT = 7
LW = 1.1

FIG2_X = ([0.8, 2.3, 4.0, 1.0], [2.5, 0.5, 0.9, -1.0])   # positions, tangents
FIG2_Y = ([1.0, 3.6, 2.0, 2.0], [1.0, -1.0, 7.0, -1.0])
FIG2_Q = (-0.98, -0.6, 1.0, 2.6, 2.98, 2.9999)
FIG2_RANGE = {-0.6: ((-1, 5), (-1, 4)), 1.0: ((-1, 5), (-1, 4)), 2.6: ((-1, 5), (-1, 4))}
FIG1 = {  # q: (t-range, y-range)  -- from the notebook's PlotRange options
    -0.8: ((-1.2, 1.2), (-1.9, 1.8)),
    0.8: ((-4.0, 4.0), (-3.0, 3.0)),
    1.0: ((-3.0, 2.5), (-3.0, 3.0)),
    2.5: ((-2.5, 2.5), (-2.2, 2.1)),
}
FIG5_A, FIG5_NP, FIG5_QGLOBAL = 3, 7, 2.6
FIG5_QSEG = (2.9, 1.4, -0.2, 2.8, 2.1, -0.5)


def qtag(q: float) -> str:
    """File-name tag for q without dots/commas: -0.8 -> 'm0p8', 2.9999 -> '2p9999'."""
    return f"{q:g}".replace("-", "m").replace(".", "p")


def _rc(plt):
    plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 1,
                         "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "mathtext.fontset": "cm",
                         "lines.solid_capstyle": "round",
                         "pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3


def _axes_origin(ax, xlabel, ylabel, grid=False):
    """Mathematica-like axes: through the origin (clamped), labels at the ends."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.spines["left"].set_position(("data", min(max(0.0, x0), x1)))
    ax.spines["bottom"].set_position(("data", min(max(0.0, y0), y1)))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ya = min(max(0.0, y0), y1)
    xa = min(max(0.0, x0), x1)
    x_top = (y1 - ya) < 0.1 * (y1 - y0)          # x axis at the top edge
    y_right = (x1 - xa) < 0.1 * (x1 - x0)        # y axis at the right edge
    if y_right:
        ax.annotate(xlabel, xy=(x0, ya), xytext=(0, -12 if x_top else 2), textcoords="offset points",
                    ha="left", va="top" if x_top else "bottom", annotation_clip=False)
    else:
        ax.annotate(xlabel, xy=(x1, ya), xytext=(-1, -3 if x_top else 2), textcoords="offset points",
                    ha="right" if x_top else "left", va="top" if x_top else "bottom", annotation_clip=False)
    if x0 < 0 < x1 and y0 < 0 < y1:   # axes cross inside: no '0' labels at the origin (as Mathematica)
        from matplotlib.ticker import FuncFormatter
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-12 else f"{v:g}".replace("-", "\u2212")))
    ax.annotate(ylabel, xy=(xa, y0 if x_top else y1), xytext=(-3 if y_right else 3, 2 if x_top else 0), textcoords="offset points",
                ha="right" if y_right else "left", va="bottom" if x_top else "top", annotation_clip=False)
    if grid:
        ax.grid(True, color="#bdbdbd", lw=0.5)
        ax.set_axisbelow(True)


def _real_power(base, e):
    """base**e where it is real: all t for an integer exponent (analytic
    continuation, as plotted by Mathematica), else only where base > 0."""
    if abs(e - round(e)) < 1e-12:
        return base ** int(round(e))
    out = np.full_like(base, np.nan)
    m = base > 0
    out[m] = base[m] ** e
    return out


def _basis_plot(t, q):
    if q == 1.0:
        return np.ones_like(t), t, t ** 2 - 1, t ** 3 - 3 * t
    h0 = _real_power(1 - (q - 1) ** 2 * t ** 4 / 4, 1 / (1 - q))
    # positive-part convention of Eq. (q_exponential): the basis lives on I_q only, so the
    # analytic continuation that Mathematica draws for integer exponents (q = 0.8) is cut
    h0 = np.where(1 - (q - 1) ** 2 * t ** 4 / 4 > 0, h0, np.nan)
    k1, k2, k3 = (3 - q) / 2, (q - 3) * (q - 2) / 2, 4 - q
    return h0, t * h0, (k1 * t ** 2 - 1) * h0, (k2 * t ** 3 - k3 * t) * h0


def _w(a, x, K=250):
    k = np.arange(1, K + 1, dtype=float)[:, None] ** a
    return np.sum(np.sin(np.pi * k * np.asarray(x)[None, :]) / (np.pi * k), axis=0)


def _clip_jumps(y, lim):
    """Break the line where it leaves the plot range (avoids vertical joins)."""
    y = np.array(y, dtype=float)
    lo, hi = lim
    span = hi - lo
    y[(y < lo - 5 * span) | (y > hi + 5 * span)] = np.nan
    return y


# Floating labels of Fig. 1 (as in the original Mathematica figures): for each
# h_j, the point t on the curve and the label position (None = on the curve;
# otherwise a thin leader joins label and curve).
FIG1_LABELS = {
    -0.8: ((0.45, None), (0.45, None), (-0.15, None), (0.78, None)),
    0.8: ((0.6, (1.3, 2.5)), (-1.3, (-0.9, -2.4)), (-2.0, (-2.7, 2.4)), (0.95, (1.6, -2.5))),
    1.0: ((-2.4, None), (-1.0, None), (-1.9, None), (1.35, None)),
    2.5: ((0.35, (1.75, 1.45)), (0.55, (1.75, 0.45)), (0.6, (1.75, -1.0)), (-0.45, (-1.75, -0.6))),
}


def _float_labels(ax, q, hs_at):
    for j, (tc, txy) in enumerate(FIG1_LABELS[q]):
        yc = float(hs_at(tc)[j])
        box = dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", lw=0.5)
        lab = rf"$H^3_{{q,{j}}}(t)$"
        if txy is None:
            ax.annotate(lab, (tc, yc), ha="center", va="center", fontsize=FONT - 1, zorder=10, bbox=box)
        else:
            ax.annotate(lab, (tc, yc), xytext=txy, textcoords="data", ha="center", va="center",
                        fontsize=FONT - 1, zorder=10, bbox=box,
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=0, shrinkB=1))


def fig1_basis(plt, outdir):
    for q, (tr, yr) in FIG1.items():
        t = np.linspace(tr[0], tr[1], 4001)
        hs = _basis_plot(t, q)
        fig, ax = plt.subplots(figsize=(2.4, 2.1))
        for j, (h, c) in enumerate(zip(hs, MMA)):
            ax.plot(t, _clip_jumps(h, yr), "-", color=c, lw=LW)
        ax.set_xlim(*tr)
        ax.set_ylim(*yr)
        _float_labels(ax, q, lambda tc, q=q: [float(v[0]) for v in _basis_plot(np.array([tc]), q)])
        _axes_origin(ax, "$t$", "$f(t)$")
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(outdir, f"fig1_basis_q{qtag(q)}.pdf"))
        plt.close(fig)


def fig2_parametric(plt, outdir):
    t = np.linspace(0, 1, 1500)
    for q in FIG2_Q:
        fig, ax = plt.subplots(figsize=(1.75, 1.6))
        xs, ys = [], []
        for i in range(3):
            x = xhermite_segment(t, q, FIG2_X[0][i], FIG2_X[0][i + 1], FIG2_X[1][i], FIG2_X[1][i + 1])
            y = xhermite_segment(t, q, FIG2_Y[0][i], FIG2_Y[0][i + 1], FIG2_Y[1][i], FIG2_Y[1][i + 1])
            ax.plot(x, y, "-", color=MMA[i], lw=LW)
            xs.append(x)
            ys.append(y)
        if q in FIG2_RANGE:
            (xa, xb), (ya, yb) = FIG2_RANGE[q]
        else:  # automatic range, as in the notebook (PlotRange -> Automatic)
            xa, xb = float(np.min(xs)), float(np.max(xs))
            ya, yb = float(np.min(ys)), float(np.max(ys))
            dx, dy = 0.04 * (xb - xa), 0.04 * (yb - ya)
            xa, xb, ya, yb = xa - dx, xb + dx, ya - dy, yb + dy
        ax.set_xlim(xa, xb)
        ax.set_ylim(ya, yb)
        if q == 1.0:
            ax.plot(FIG2_X[0], FIG2_Y[0], "o", color="#333333", ms=2.5, zorder=5)
            for j, (px, py) in enumerate(zip(FIG2_X[0], FIG2_Y[0]), start=1):
                ax.annotate(f"{j}", (px, py), textcoords="offset points", xytext=(0, 4),
                            fontsize=FONT - 1, ha="center", va="bottom")
        _axes_origin(ax, "$x_q(t)$", "$y_q(t)$", grid=True)
        ax.tick_params(labelsize=FONT - 2)
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(outdir, f"fig2_curve_q{qtag(q)}.pdf"))
        plt.close(fig)


def fig3_weierstrass(plt, outdir):
    x = np.linspace(0, 1, 6000)
    fig, ax = plt.subplots(figsize=(4.7, 2.3))
    for a, c in zip((2, 3, 4, 5), MMA):
        ax.plot(x, _w(a, x), "-", color=c, lw=0.6, label=f"$W_{a}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(frameon=True, edgecolor="#999999", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _axes_origin(ax, "$x$", "$W_a(x)$")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(outdir, "fig3_weierstrass.pdf"))
    plt.close(fig)


def _piecewise(ax, P, xs, q_per_seg, a, c_class, c_q, lw_class, mid=False):
    n = len(P) - 1
    x = np.linspace(xs[0], xs[-1], 4000)
    L = xs[-1] - xs[0]
    ax.plot(x, _w(a, (x - xs[0]) / L), "-", color=C_W, lw=0.5, label=f"$W_{a}$", zorder=2)
    t = np.linspace(0, 1, 300)
    for i in range(n):
        xx = xs[i] + t * (xs[i + 1] - xs[i])
        ax.plot(xx, xhermite_segment(t, 1.0, P[i], P[i + 1]), "-", color=c_class, lw=lw_class, zorder=3,
                label="classical Hermite ($q=1$)" if i == 0 else None)
        ax.plot(xx, xhermite_segment(t, q_per_seg[i], P[i], P[i + 1]), "-", color=c_q, lw=LW, zorder=4,
                label="x-Hermite" if i == 0 else None)
    return x


def fig4_constants(plt, outdir):
    n, a = 12, 3
    P = [weierstrass_rational(a, i, n) for i in range(n + 1)]
    xs = np.arange(n + 1, dtype=float)
    for q in (2.9, -0.6):
        fig, ax = plt.subplots(figsize=(2.4, 2.25))
        ax.vlines(xs, 0, P, color=C_CLASS_F4, lw=0.6, zorder=1)
        _piecewise(ax, P, xs, [q] * n, a, C_CLASS_F4, C_Q_F4, LW)
        ax.plot(xs, P, "o", color="black", ms=2.8, zorder=6)
        ax.set_xlim(0, 12.3)
        ax.set_ylim(0, 0.5)
        ax.set_title(f"$q={q:g}$", fontsize=FONT)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
                  handlelength=1.4, columnspacing=0.8, fontsize=FONT - 1.5)
        _axes_origin(ax, "$x$", "$f(x)$")
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(outdir, f"fig4_constants_q{qtag(q)}.pdf"))
        plt.close(fig)


def fig5_local_q(plt, outdir):
    ni = FIG5_NP - 1
    P = [weierstrass_rational(FIG5_A, i, ni) for i in range(FIG5_NP)]
    xs = np.arange(FIG5_NP) / ni
    xm = (xs[:-1] + xs[1:]) / 2
    for tag, qs, title in (("a_global", [FIG5_QGLOBAL] * ni, f"$q={FIG5_QGLOBAL:g}$"),
                           ("b_per_segment", list(FIG5_QSEG), "one $q$ per interval")):
        fig, ax = plt.subplots(figsize=(2.4, 2.25))
        Pm = [float(xhermite_segment(0.5, qs[i], P[i], P[i + 1])) for i in range(ni)]
        ax.vlines(np.r_[xs, xm], 0, np.r_[P, Pm], color="#9fa8e0", lw=0.5, zorder=1)
        _piecewise(ax, P, xs, qs, FIG5_A, C_CLASS_F5, C_Q_F5, 0.9)
        ax.plot(xm, Pm, "o", color=C_Q_F5, ms=2.8, zorder=6, label="new samples")
        ax.plot(xs, P, "o", color="black", ms=2.8, zorder=6)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 0.4)
        ax.set_title(title, fontsize=FONT)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2,
                  handlelength=1.4, columnspacing=0.8, fontsize=FONT - 1.5)
        _axes_origin(ax, "$x$", "$f(x)$")
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(outdir, f"fig5{tag}.pdf"))
        plt.close(fig)


def build_figures(outdir: str) -> list:
    """Write Figs. 1-5 to <outdir>/figuras and return the list of files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = os.path.join(outdir, "figuras")
    os.makedirs(fdir, exist_ok=True)
    _rc(plt)
    for f in (fig1_basis, fig2_parametric, fig3_weierstrass, fig4_constants, fig5_local_q):
        f(plt, fdir)
    return sorted(x for x in os.listdir(fdir) if x.startswith(("fig1_", "fig2_", "fig3_", "fig4_", "fig5")))


if __name__ == "__main__":
    import sys
    print("\n".join(build_figures(sys.argv[1] if len(sys.argv) > 1 else ".")))
