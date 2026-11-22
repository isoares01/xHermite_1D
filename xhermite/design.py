"""
Geometric (design-oriented) illustration of the x-Hermite segment: invariances of the basic
form and of the level-relative variants (Section 4.4 of the article).

``python -m xhermite.design --outdir <article dir>`` writes

    figuras/fig_design.pdf       3 panels: (a) fixed Hermite data, several q (centered variant);
                                 (b) translation test; (c) reversal test
    results/macros_design.tex    the numbers quoted in Section 4.4 (macros ``\\dsg*``)
    results/design_report.txt    the same numbers in plain text

Planar test data (positions and tangents in the local parameter): an arch,
P0 = (1, 1), Pf = (5, 1.5), V0 = (3, 5), Vf = (3, -4). All deviations are
reported relative to the largest extent of the classical (q = 1) segment.

Four schemes are compared:
    raw      basic x-Hermite segment (Section 3), anchored at the coordinate origin;
    left     left-anchored level-relative variant  P0 + s[0, Pf-P0, V0, Vf];
    midraw   anchored at the mean level m = (P0+Pf)/2 but with the ORIGINAL weight w_q
             (shows that the even weight is also needed for reversal symmetry);
    centered centered level-relative variant (Proposition 11): mean level and even weight.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .mathematica_reference import xhermite_segment
from .variants import centered_segment, left_anchored_segment

P0 = np.array([1.0, 1.0])
P1 = np.array([5.0, 1.5])
V0 = np.array([3.0, 5.0])
V1 = np.array([3.0, -4.0])
T = np.linspace(0.0, 1.0, 2001)
TRANSLATION = np.array([10.0, -7.0])


def _midraw(t, q, p0, p1, v0, v1):
    d = p1 - p0
    return 0.5 * (p0 + p1) + xhermite_segment(t, q, -0.5 * d, 0.5 * d, v0, v1)


SCHEMES = {"raw": xhermite_segment, "left": left_anchored_segment, "midraw": _midraw, "centered": centered_segment}


def curve(scheme, q, p0=P0, p1=P1, v0=V0, v1=V1, t=T):
    """Planar curve (len(t), 2) of the given scheme, coordinatewise."""
    fn = SCHEMES[scheme]
    return np.stack([np.asarray(fn(t, q, p0[k], p1[k], v0[k], v1[k]), dtype=float) for k in range(2)], axis=1)


def size_of(c):
    return float(np.ptp(c, axis=0).max())


_SIZE = size_of(curve("centered", 1.0))


def rel(x):
    """Max Euclidean norm of the rows of x, relative to the classical extent."""
    return float(np.max(np.linalg.norm(x, axis=1)) / _SIZE)


def translation_deviation(scheme, q, c=TRANSLATION):
    """Shape change caused by translating the data by c: max |s[R+c] - c - s[R]| (relative)."""
    return rel(curve(scheme, q, P0 + c, P1 + c) - c - curve(scheme, q))


def reversal_deviation(scheme, q):
    """max |s[R^rev](1-t) - s[R](t)| (relative); R^rev = (Pf, P0, -Vf, -V0)."""
    return rel(curve(scheme, q, P1, P0, -V1, -V0)[::-1] - curve(scheme, q))


def deviation_from_classical(q, scheme="centered"):
    return rel(curve(scheme, q) - curve("centered", 1.0))


def blending_deviation(q, scheme):
    """max_j sup_t |phi_j - psi_j| for the unit-data blending functions of the scheme."""
    t = T
    psi = [lambda t: 2 * t ** 3 - 3 * t ** 2 + 1, lambda t: -2 * t ** 3 + 3 * t ** 2,
           lambda t: t ** 3 - 2 * t ** 2 + t, lambda t: t ** 3 - t ** 2]
    fn = SCHEMES[scheme]
    out = []
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1.0
        out.append(float(np.max(np.abs(np.asarray(fn(t, q, *e), dtype=float) - psi[j](t)))))
    return max(out)


def tangent_rescaling_residual(q):
    """Fraction of the deviation from the classical centered segment that remains after the best
    classical cubic Hermite with rescaled end tangents (alpha*V0, beta*Vf) is subtracted."""
    from scipy.optimize import minimize
    target = curve("centered", q)
    dev = np.max(np.linalg.norm(target - curve("centered", 1.0), axis=1))

    def err(ab):
        c = curve("centered", 1.0, v0=ab[0] * V0, v1=ab[1] * V1)
        return float(np.max(np.linalg.norm(c - target, axis=1)))

    best = min((minimize(err, x0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12))
                for x0 in ([1.0, 1.0], [0.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.5, 0.5])), key=lambda r: r.fun)
    return float(best.fun / dev)


def numbers():
    """All numbers quoted in the article (dictionary name -> value)."""
    qs = np.round(np.arange(0.5, 1.5001, 0.01), 2)
    res = {}
    res["DevHalf"] = 100 * deviation_from_classical(0.5)
    res["DevOneHalf"] = 100 * deviation_from_classical(1.5)
    res["DevTwoHalf"] = 100 * deviation_from_classical(2.5)
    res["DevTwoNine"] = 100 * deviation_from_classical(2.9)
    res["BlendCen"] = 100 * max(blending_deviation(q, "centered") for q in qs)
    res["BlendRaw"] = 100 * max(blending_deviation(q, "raw") for q in qs)
    r = [tangent_rescaling_residual(q) for q in (0.5, 1.5, 2.5)]
    res["TangResMin"], res["TangResMax"] = 100 * min(r), 100 * max(r)
    res["TransRawHalf"] = 100 * translation_deviation("raw", 0.5)
    res["TransRawTwo"] = 100 * translation_deviation("raw", 2.0)
    res["TransLeft"] = max(translation_deviation("left", q) for q in (0.5, 2.0))
    res["TransCen"] = max(translation_deviation("centered", q) for q in (0.5, 2.0))
    res["RevRawTwo"] = 100 * reversal_deviation("raw", 2.0)
    res["RevLeftTwo"] = 100 * reversal_deviation("left", 2.0)
    res["RevMidRawTwo"] = 100 * reversal_deviation("midraw", 2.0)
    res["RevCen"] = max(reversal_deviation("centered", q) for q in (0.5, 2.0))
    res["TransNorm"] = float(np.linalg.norm(TRANSLATION))
    return res


def _fmt(name, v):
    if name in ("TransLeft", "TransCen", "RevCen"):       # rounding-level quantities
        m, e = f"{v:.0e}".split("e")
        return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"
    if name == "TransNorm":
        return f"{v:.1f}"
    return f"{v:.1f}"


def write_macros(outdir, res):
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    with open(os.path.join(outdir, "results", "macros_design.tex"), "w") as fh:
        fh.write("% generated by `python -m xhermite.design` -- do not edit by hand\n")
        for k, v in res.items():
            fh.write(f"\\newcommand{{\\dsg{k}}}{{{_fmt(k, v)}}}\n")
    with open(os.path.join(outdir, "results", "design_report.txt"), "w") as fh:
        fh.write("Design-oriented checks (relative to the extent of the classical segment, in %)\n")
        for k, v in res.items():
            fh.write(f"{k:14s} {v:.6g}\n")


def write_figure(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import MMA, _rc

    _rc(plt)
    os.makedirs(os.path.join(outdir, "figuras"), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.45), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) fixed Hermite data, several q (centered variant)
    ax = axs[0]
    qs = [(-0.5, MMA[1]), (0.5, MMA[2]), (1.0, "black"), (2.0, MMA[3]), (2.5, MMA[0])]
    for q, col in qs:
        c = curve("centered", q)
        ax.plot(c[:, 0], c[:, 1], color=col, lw=1.0, label=f"$q={q:g}$" + (" (classical)" if q == 1 else ""))
    for P, V, s in ((P0, V0, 1), (P1, V1, 1)):
        ax.annotate("", xy=P + V / 3, xytext=P, arrowprops=dict(arrowstyle="->", color="#6b6b6b", lw=0.7))
        ax.plot(*P, "o", color="black", ms=3.2, zorder=5)
    ax.set_xlim(0.4, 6.4)
    ax.set_ylim(-1.6, 3.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$", labelpad=1)
    ax.set_ylabel("$y$", labelpad=1)
    ax.legend(loc="lower center", ncol=2, frameon=False, handlelength=1.3, columnspacing=0.8, borderaxespad=0.1)
    ax.set_title("(a) centered variant, fixed Hermite data", fontsize=7, pad=3)

    # (b) translation test
    ax = axs[1]
    norms = np.linspace(0.0, 20.0, 41)
    direction = TRANSLATION / np.linalg.norm(TRANSLATION)
    for (scheme, q, col, ls, lab) in (("raw", 0.5, MMA[2], "-", "basic, $q=0.5$"),
                                       ("raw", 2.0, MMA[3], "-", "basic, $q=2$"),
                                       ("centered", 2.0, "black", "--", "centered")):
        y = [100 * translation_deviation(scheme, q, s * direction) for s in norms]
        ax.plot(norms, y, ls, color=col, lw=1.0, label=lab)
    ax.set_xlabel("translation length $\\|c\\|$", labelpad=1)
    ax.set_ylabel("shape change (% of extent)", labelpad=1)
    ax.set_title("(b) translating the data by $c$", fontsize=7, pad=3)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5, borderaxespad=0.1)
    ax.grid(True, color="#e6e6e6", lw=0.5)

    # (c) reversal test
    ax = axs[2]
    qg = np.linspace(-0.5, 2.7, 65)
    for scheme, col, ls, lab in (("raw", MMA[3], "-", "basic"),
                                 ("left", MMA[1], "-", "left-anchored"),
                                 ("midraw", "#e08a00", (0, (1, 1.2)), "mean anchor, weight $w_q$"),
                                 ("centered", "black", "--", "centered")):
        ax.plot(qg, [100 * reversal_deviation(scheme, q) for q in qg], ls=ls, color=col, lw=1.0, label=lab)
    ax.set_xlabel("$q$", labelpad=1)
    ax.set_ylabel("reversal defect (% of extent)", labelpad=1)
    ax.set_ylim(-0.4, 9)
    ax.set_title("(c) reversing the orientation", fontsize=7, pad=3)
    ax.legend(loc="upper center", frameon=False, handlelength=1.8, borderaxespad=0.1)
    ax.grid(True, color="#e6e6e6", lw=0.5)

    for a in axs[1:]:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    path = os.path.join(outdir, "figuras", "fig_design.pdf")
    fig.savefig(path)
    plt.close(fig)
    plt.rcdefaults()
    return path


def build(outdir):
    res = numbers()
    write_macros(outdir, res)
    return write_figure(outdir), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".", help="article directory (default: current)")
    a = ap.parse_args(argv)
    path, res = build(a.outdir)
    print(path)
    for k, v in res.items():
        print(f"{k:14s} {v:.6g}")


if __name__ == "__main__":
    main()
