"""
Behaviour of the x-Hermite segment near the four exceptional values of q
(Section 3.5 of the article): q = -1, 1, 2 and 3.

``python -m xhermite.singularity --outdir <article dir>`` writes

    figuras/fig_singular.pdf        2 x 4 panels, one column per exceptional value
    figuras/fig_extent.pdf          size of the planar curve of Fig. 2 versus the distance to q = -1, 3
    results/macros_singular.tex     the numbers quoted in the text (macros ``\\sg*``)
    results/singular_report.txt     the same numbers in plain text

What is measured
----------------
Intrinsic conditioning (row 1; independent of any algorithm), for the segment
s_q = sum_j D_j phi_{j,q}(t) with data D = (P0, Pf, V0, Vf):

    gain(q)  = sup_t sum_j |phi_{j,q}(t)|                 amplification of the data
    kappa(q) = |q| sup_t |d s_q / d q| / sup_t |s_q|      relative condition number
                                                          with respect to the parameter q
                                                          (test data D_TEST)

Floating-point error (row 2), relative max-norm error of the curve against a
60-digit evaluation of the factorized form (Theorem 2):

    closed  : Appendix A closed form of M_q, R . M_q . H_q(t), in double precision;
    factor  : factorized evaluator ``xhermite_segment`` (base 1 - x through log1p);
    product : factorized evaluator with the base written as (1 - u)(1 + u), u = (q-1) t^2 / 2,
              and each factor as a sum of non-negative terms, which removes the cancellation
              at t = 1 near q = -1 and q = 3 (the dotted line in the figure is then the
              unavoidable error eps * kappa_q caused by a rounding error eps in q itself).

Leading terms at the genuine singularities (derived in the article, Theorem 3):
    q -> 3^- , d = 3 - q :  a_q ~ d^(-1/2),  b_q / a_q^2 ~ 2 d^(-1/2),
                            phi_1 ~ 2 d^(-1/2) t^2 (1-t) (1-t^4)^(-1/2)
    q -> -1^+, d = 1 + q :  a_q ~ d^(1/2),   b_q / a_q^2 ~ -2 d^(-3/2),
                            phi_1 ~ -2 d^(-3/2) t^2 (1-t) (1-t^4)^(1/2)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mpmath as mp

from .basis import compute_Mq_matrix_closed, q_hermite_basis_numeric
from .mathematica_reference import xhermite_segment

mp.mp.dps = 60

D_TEST = (1.0, 2.5, 0.5, -1.0)          # generic data (P0, Pf, V0, Vf)
T_GRID = np.linspace(0.0, 1.0, 401)
EPS = np.finfo(float).eps

# Planar curve of Fig. 2: (x, x'; y, y') at the four nodes (tangents in the local parameter)
FIG2_NODES = [((0.8, 2.5), (1.0, 1.0)), ((2.3, 0.5), (3.6, -1.0)),
              ((4.0, 0.9), (2.0, 7.0)), ((1.0, -1.0), (2.0, -1.0))]

EXCEPTIONAL = (-1.0, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# High-precision reference
# ---------------------------------------------------------------------------
def _mpq(q):
    return mp.mpf(q)


def mp_weight(t, q):
    """w_q(t) and w_q'(t), exact for -1 < q < 3, q != 1 (and the limit w = 1 at q = 1)."""
    q, t = _mpq(q), mp.mpf(t)
    if q == 1:
        return mp.mpf(1), mp.mpf(0)
    base = 1 - (q - 1) ** 2 * t ** 4 / 4
    return base ** (1 / (1 - q)), (q - 1) * t ** 3 * base ** (q / (1 - q))


def mp_phi(t, q):
    """(phi_0, ..., phi_3) of Eq. (blending) at t, 60 digits."""
    t = mp.mpf(t)
    w, _ = mp_weight(t, q)
    a, b = mp_weight(1, q)
    psi00 = 2 * t ** 3 - 3 * t ** 2 + 1
    psi01 = -2 * t ** 3 + 3 * t ** 2
    psi10 = t ** 3 - 2 * t ** 2 + t
    psi11 = t ** 3 - t ** 2
    return (w * psi00, w * (psi01 / a - b / a ** 2 * psi11), w * psi10, w * psi11 / a)


def mp_curve(q, data=D_TEST, t=T_GRID):
    return np.array([float(sum(d * f for d, f in zip(data, mp_phi(tt, q)))) for tt in t])


def gain(q, t=T_GRID):
    """sup_t sum_j |phi_{j,q}(t)| (60-digit evaluation)."""
    return float(max(sum(abs(f) for f in mp_phi(tt, q)) for tt in t))


def sup_phi(q, j, t=T_GRID):
    return float(max(abs(mp_phi(tt, q)[j]) for tt in t))


def kappa(q, data=D_TEST, t=T_GRID):
    """|q| sup_t |ds/dq| / sup_t |s|: relative condition number of the segment with respect to q."""
    q = mp.mpf(q)
    h = mp.mpf(10) ** (-25) * max(1, abs(q))
    num = 0
    den = 0
    for tt in t:
        sp = sum(d * f for d, f in zip(data, mp_phi(tt, q + h)))
        sm = sum(d * f for d, f in zip(data, mp_phi(tt, q - h)))
        s0 = sum(d * f for d, f in zip(data, mp_phi(tt, q)))
        num = max(num, abs((sp - sm) / (2 * h)))
        den = max(den, abs(s0))
    return float(abs(q) * num / den)


# ---------------------------------------------------------------------------
# Double-precision evaluators
# ---------------------------------------------------------------------------
def seg_closed(q, data=D_TEST, t=T_GRID):
    """R . M_q . H_q(t) with the Appendix A matrix; NaN where the formula overflows."""
    try:
        with np.errstate(all="ignore"):
            M = compute_Mq_matrix_closed(q)
            H = np.array(q_hermite_basis_numeric(t, q))
            out = np.array(data) @ M @ H
        return out
    except (OverflowError, ZeroDivisionError):
        return np.full_like(t, np.nan)


def seg_factor(q, data=D_TEST, t=T_GRID):
    with np.errstate(all="ignore"):
        return np.array([float(xhermite_segment(tt, q, *data)) for tt in t])


def seg_product(q, data=D_TEST, t=T_GRID):
    """Factorized form with the base 1 - (q-1)^2 t^4/4 computed as (1-u)(1+u), u = (q-1) t^2/2."""
    e = q - 1.0
    t = np.asarray(t, dtype=float)

    def lw(tt):
        tt = np.asarray(tt, dtype=float)
        t2 = tt * tt
        u = e * t2 / 2.0
        # 1 - u = (1 - t^2) + (3 - q) t^2 / 2 and 1 + u = (1 - t^2) + (q + 1) t^2 / 2 are sums of
        # non-negative terms for -1 < q < 3: no cancellation anywhere, also at t = 1, q -> -1, 3.
        prod = ((1.0 - t2) + (3.0 - q) * t2 / 2.0) * ((1.0 - t2) + (q + 1.0) * t2 / 2.0)
        # near q = 1 (|u| small) log1p(-u^2) is the accurate way to take the logarithm
        return np.where(np.abs(u) < 0.5, np.log1p(-u * u), np.log(prod))

    if e == 0.0:
        w = np.ones_like(t)
        a, b = 1.0, 0.0
    else:
        w = np.exp(lw(t) / (1.0 - q))
        a = float(np.exp(lw(1.0) / (1.0 - q)))
        b = float(e * np.exp(q / (1.0 - q) * lw(1.0)))
    P0, P1, V0, V1 = data
    p1 = P1 / a
    m1 = (V1 - b * p1) / a
    H = (P0 * (2 * t ** 3 - 3 * t ** 2 + 1) + p1 * (-2 * t ** 3 + 3 * t ** 2)
         + V0 * (t ** 3 - 2 * t ** 2 + t) + m1 * (t ** 3 - t ** 2))
    return w * H


def rel_err(s, ref):
    with np.errstate(all="ignore"):
        e = float(np.max(np.abs(s - ref)) / np.max(np.abs(ref)))
    return e if np.isfinite(e) else np.nan


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def sweep(q0, dmin=1e-12, dmax=1e-1, n=34):
    """Values at q = q0 -+ d (both sides at the interior points q0 = 1, 2)."""
    ds = np.logspace(np.log10(dmin), np.log10(dmax), n)
    sides = (1.0,) if q0 == -1.0 else (-1.0,) if q0 == 3.0 else (-1.0, 1.0)
    rows = []
    for s in sides:
        for d in ds:
            q = q0 + s * d
            ref = mp_curve(q)
            rows.append(dict(q0=q0, side=s, d=d, q=q, gain=gain(q), kappa=kappa(q),
                             closed=rel_err(seg_closed(q), ref),
                             factor=rel_err(seg_factor(q), ref),
                             product=rel_err(seg_product(q), ref)))
    return rows


def fitted_slope(rows, dlo=1e-6, dhi=1e-3, key="gain"):
    d = np.array([r["d"] for r in rows if dlo <= r["d"] <= dhi])
    y = np.array([r[key] for r in rows if dlo <= r["d"] <= dhi])
    return float(np.polyfit(np.log(d), np.log(y), 1)[0])


def asymptotic_constants():
    """sup of the limit profiles of phi_1 (see module docstring)."""
    t = np.linspace(0, 1, 200001)
    c3 = 2 * np.max(t ** 2 * (1 - t) / np.sqrt((1 - t ** 4) + 1e-300) * (t < 1))
    cm1 = 2 * np.max(t ** 2 * (1 - t) * np.sqrt(1 - t ** 4))
    return float(c3), float(cm1)


def extent(q):
    c = np.concatenate(curve_extent_q(q))
    return float(np.ptp(c, axis=0).max()), float(np.abs(c).max())


def curve_extent_q(q):
    t = np.linspace(0, 1, 801)
    out = []
    for i in range(3):
        cols = []
        for k in range(2):
            p0 = FIG2_NODES[i][k][0]
            p1 = FIG2_NODES[i + 1][k][0]
            v0 = FIG2_NODES[i][k][1]
            v1 = FIG2_NODES[i + 1][k][1]
            cols.append(seg_product(q, (p0, p1, v0, v1), t))
        out.append(np.stack(cols, axis=1))
    return out


def lipschitz_estimate(qlo, qhi, nq=121):
    """Sampled L_Q = sup_{q in Q} max_j sup_t |phi_j - psi_j| / |q-1| (estimate, not a bound)."""
    t = np.linspace(0, 1, 201)
    qs = np.linspace(qlo, qhi, nq)
    qs = qs[np.abs(qs - 1) > 1e-9]
    best = 0.0
    for q in qs:
        for tt in t:
            ph = mp_phi(tt, q)
            tt_ = mp.mpf(tt)
            ps = (2 * tt_ ** 3 - 3 * tt_ ** 2 + 1, -2 * tt_ ** 3 + 3 * tt_ ** 2,
                  tt_ ** 3 - 2 * tt_ ** 2 + tt_, tt_ ** 3 - tt_ ** 2)
            best = max(best, float(max(abs(a - b) for a, b in zip(ph, ps)) / abs(mp.mpf(q) - 1)))
    return best


# ---------------------------------------------------------------------------
# Numbers, macros, figures
# ---------------------------------------------------------------------------
def numbers(fast=False):
    res = {}
    n = 18 if fast else 34
    sweeps = {q0: sweep(q0, n=n) for q0 in EXCEPTIONAL}
    res["_sweeps"] = sweeps
    res["slopeThree"] = fitted_slope(sweeps[3.0])
    res["slopeMinus"] = fitted_slope(sweeps[-1.0])
    res["kappaSlopeThree"] = fitted_slope(sweeps[3.0], key="kappa")
    res["kappaSlopeMinus"] = fitted_slope(sweeps[-1.0], key="kappa")
    c3, cm = asymptotic_constants()
    res["cThree"], res["cMinus"] = c3, cm
    for q, tag in ((2.9999, "ThreeFour"), (2.98, "TwoNineEight"), (2.6, "TwoSix"),
                   (-0.98, "MinusZeroNineEight"), (-0.6, "MinusZeroSix")):
        a, b = (float(v) for v in mp_weight(1, q))
        res["a" + tag] = a
        res["ba" + tag] = b / a ** 2
        s1, m1 = extent(1.0)
        s, m = extent(q)
        res["ext" + tag] = s / s1
        res["max" + tag] = m
    for q, tag in ((1.0, "One"), (-0.8, "MinusZeroEight"), (2.9, "TwoNine")):
        res["gain" + tag] = gain(q)
    res["maxClassical"] = extent(1.0)[1]
    res["sizeClassical"] = extent(1.0)[0]
    # rates of a_q: exact limits of a_q d^{1/2} (q->3) and a_q d^{-1/2} (q->-1)
    res["aLimThree"] = float(mp_weight(1, 3 - 1e-10)[0]) * 1e-5
    res["aLimMinus"] = float(mp_weight(1, -1 + 1e-10)[0]) * 1e5
    # error floor of the evaluators at the closest points
    for q0, tag in ((1.0, "One"), (2.0, "Two"), (-1.0, "Minus"), (3.0, "Three")):
        near = min(sweeps[q0], key=lambda r: abs(np.log10(r["d"]) + 8))
        res["errClosed" + tag] = near["closed"]
        res["errFactor" + tag] = near["factor"]
        res["errProduct" + tag] = near["product"]
    if not fast:
        for (lo, hi), tag in (((0.5, 1.5), "Half"), ((0.0, 2.0), "ZeroTwo"), ((-0.5, 2.5), "MidWide"),
                              ((-0.8, 2.9), "Wide"), ((-0.95, 2.95), "EdgeA"), ((-0.99, 2.99), "EdgeB")):
            res["L" + tag] = lipschitz_estimate(lo, hi, nq=61)
    return res


def _fmt(name, v):
    if v != v:  # NaN: the closed form overflowed
        return f"\\newcommand{{\\sg{name}}}{{overflow}}"
    if isinstance(v, float) and (abs(v) >= 1e4 or (0 < abs(v) < 1e-3)):
        m, e = f"{v:.2e}".split("e")
        return f"\\newcommand{{\\sg{name}}}{{\\ensuremath{{{m}\\times10^{{{int(e)}}}}}}}"
    if name.startswith("kappaSlope"):  # exponents close to -1: keep 3 decimals (-1.000, -1.001)
        return f"\\newcommand{{\\sg{name}}}{{{v:.3f}}}"
    return f"\\newcommand{{\\sg{name}}}{{{v:.3g}}}"


def write_macros(outdir, res):
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    with open(os.path.join(outdir, "results", "macros_singular.tex"), "w") as fh:
        fh.write("% generated by python -m xhermite.singularity -- do not edit\n")
        for k, v in res.items():
            if k.startswith("_"):
                continue
            fh.write(_fmt(k, float(v)) + "%\n")
    with open(os.path.join(outdir, "results", "singular_report.txt"), "w") as fh:
        fh.write("Stability at the exceptional values of q (see xhermite/singularity.py)\n")
        for k, v in res.items():
            if not k.startswith("_"):
                fh.write(f"{k:22s} {float(v):.6g}\n")
        for q0, rows in res["_sweeps"].items():
            fh.write(f"\n# q0 = {q0:g}: d, side, gain, kappa, err_closed, err_factor, err_product\n")
            for r in rows:
                fh.write("%.3e %+d %.3e %.3e %.2e %.2e %.2e\n" % (r["d"], r["side"], r["gain"], r["kappa"],
                                                                    r["closed"], r["factor"], r["product"]))


def write_figures(outdir, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import MMA, _rc

    _rc(plt)
    os.makedirs(os.path.join(outdir, "figuras"), exist_ok=True)
    sweeps = res["_sweeps"]
    titles = {-1.0: r"$q\to-1^{+}$ (genuine)", 1.0: r"$q\to1$ (removable)",
              2.0: r"$q\to2$ (removable)", 3.0: r"$q\to3^{-}$ (genuine)"}
    dist = {-1.0: r"$d=q+1$", 1.0: r"$d=|q-1|$", 2.0: r"$d=|q-2|$", 3.0: r"$d=3-q$"}
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.9), sharex=True)
    for c, q0 in enumerate(EXCEPTIONAL):
        rows = sweeps[q0]
        ax = axs[0, c]
        for side, ls in ((1.0, "-"), (-1.0, "--")):
            r = [x for x in rows if x["side"] == side]
            if not r:
                continue
            d = [x["d"] for x in r]
            ax.loglog(d, [x["gain"] for x in r], ls, color="black", lw=1.0,
                      label="data gain" if side == r[0]["side"] else None)
            ax.loglog(d, [x["kappa"] for x in r], ls, color=MMA[0], lw=1.0,
                      label=r"$\kappa_q$" if side == r[0]["side"] else None)
        if q0 == 3.0:
            ax.loglog([1e-10, 1e-2], [9 * 1e-10 ** -0.5 / 100 * 1e-0, 9 * 1e-2 ** -0.5 / 100], ":", color="#888888", lw=0.8)
            ax.text(2e-9, 2e2, r"$d^{-1/2}$", fontsize=6, color="#555555")
        if q0 == -1.0:
            ax.loglog([1e-10, 1e-2], [1e-10 ** -1.5, 1e-2 ** -1.5], ":", color="#888888", lw=0.8)
            ax.text(3e-9, 1e11, r"$d^{-3/2}$", fontsize=6, color="#555555")
        ax.set_title(titles[q0], fontsize=7, pad=3)
        ax.grid(True, color="#e6e6e6", lw=0.5)
        if c == 0:
            ax.set_ylabel("intrinsic sensitivity", labelpad=1)
            ax.legend(loc="upper right", frameon=False, handlelength=1.4, fontsize=6)
        ax2 = axs[1, c]
        for side, ls in ((1.0, "-"), (-1.0, "--")):
            r = [x for x in rows if x["side"] == side]
            if not r:
                continue
            d = np.array([x["d"] for x in r])
            for key, col, lab in (("closed", MMA[3], "closed form $M_q$"),
                                  ("factor", MMA[2], "factorized"),
                                  ("product", "black", "factorized, product base")):
                y = np.array([x[key] for x in r], dtype=float)
                ok = np.isfinite(y)
                ax2.loglog(d[ok], np.maximum(y[ok], 1e-17), ls, color=col, lw=1.0,
                           label=lab if side == r[0]["side"] else None)
                if (~ok).any():
                    ax2.loglog(d[~ok], np.full((~ok).sum(), 3e1), "x", color=col, ms=3)
            kap = np.array([x["kappa"] for x in r])
            ax2.loglog(d, EPS * kap, ":", color="#888888", lw=0.8,
                       label=r"$\epsilon_{\rm mach}\kappa_q$" if side == r[0]["side"] else None)
        ax2.set_xlabel(dist[q0], labelpad=1)
        ax2.set_ylim(1e-17, 1e2)
        ax2.grid(True, color="#e6e6e6", lw=0.5)
        if c == 0:
            ax2.set_ylabel("relative error (double)", labelpad=1)
            ax2.legend(loc="upper left", frameon=False, handlelength=1.4, fontsize=5.5)
        for a in (ax, ax2):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.invert_xaxis()
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    p1 = os.path.join(outdir, "figuras", "fig_singular.pdf")
    fig.savefig(p1)
    plt.close(fig)

    # size of the Fig. 2 curve versus the distance to the genuine singularities
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.2))
    d = np.logspace(-6, -0.3, 28)
    base = extent(1.0)[0]
    for ax, q0, sgn, lab, slope in ((axs[0], -1.0, 1, r"$d=q+1$", -1.5), (axs[1], 3.0, -1, r"$d=3-q$", -0.5)):
        ext = [extent(q0 + sgn * x)[0] / base for x in d]
        ax.loglog(d, ext, color="black", lw=1.0, label="curve size / classical size")
        ref = ext[len(d) // 2] * (d / d[len(d) // 2]) ** slope
        ax.loglog(d, ref, ":", color="#888888", lw=0.9, label=rf"$d^{{{slope:g}}}$")
        ax.set_xlabel(lab, labelpad=1)
        ax.invert_xaxis()
        ax.grid(True, color="#e6e6e6", lw=0.5)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("size of the Fig. 2 curve", labelpad=1)
    fig.tight_layout(pad=0.4)
    p2 = os.path.join(outdir, "figuras", "fig_extent.pdf")
    fig.savefig(p2)
    plt.close(fig)
    plt.rcdefaults()
    return p1, p2


def build(outdir, fast=False):
    res = numbers(fast=fast)
    write_macros(outdir, res)
    return write_figures(outdir, res), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".", help="article directory (default: current)")
    ap.add_argument("--fast", action="store_true", help="coarser sweeps, no Lipschitz sampling")
    a = ap.parse_args(argv)
    paths, res = build(a.outdir, fast=a.fast)
    print(*paths)
    for k, v in res.items():
        if not k.startswith("_"):
            print(f"{k:22s} {float(v):.6g}")


if __name__ == "__main__":
    main()
