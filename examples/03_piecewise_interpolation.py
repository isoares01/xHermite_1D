"""
Example 03 -- piecewise x-Hermite interpolation with a global q or one q per
segment (article, figuras/fig5*.pdf: W_3, N_P = 7 uniform nodes, zero tangents).

Usage:
    python examples/03_piecewise_interpolation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from xhermite import QHermiteSpline1D, weierstrass_function, weierstrass_rational

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

A, NP = 3, 7
Q_GLOBAL = 2.6                                   # fig5a
Q_PER_SEGMENT = [2.9, 1.4, -0.2, 2.8, 2.1, -0.5]  # fig5b


def main():
    ni = NP - 1
    x_nodes = np.arange(NP) / ni
    y_nodes = np.array([weierstrass_rational(A, i, ni) for i in range(NP)])
    spline = QHermiteSpline1D(x_nodes, y_nodes)          # dy omitted -> zero tangents

    x_mid = (np.arange(ni) + 0.5) / ni                   # new samples of a 2x upsampling
    w_mid = np.array([weierstrass_rational(A, 2 * i + 1, 2 * ni) for i in range(ni)])
    for label, q in (("classical q=1", 1.0), (f"global q={Q_GLOBAL}", Q_GLOBAL),
                     ("per-segment q", Q_PER_SEGMENT)):
        s_mid = spline.evaluate(x_mid, q=q)
        print(f"{label:>16}: RMSE at midpoints = {np.sqrt(np.mean((s_mid - w_mid) ** 2)):.5f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipping the figure)")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    x = np.linspace(0, 1, 3000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, q, title in ((axes[0], Q_GLOBAL, f"(a) q = {Q_GLOBAL}"),
                         (axes[1], Q_PER_SEGMENT, "(b) one q per segment")):
        ax.plot(x, weierstrass_function(x, A), color="#e34948", lw=0.6, label="$W_3(x)$")
        ax.plot(x, spline.evaluate(x, q=1.0), color="#2a78d6", lw=1.4, label="classical")
        ax.plot(x, spline.evaluate(x, q=q), color="#4a3aa7", lw=1.4, label="x-Hermite")
        ax.plot(x_nodes, y_nodes, "ko", ms=4)
        ax.plot(x_mid, spline.evaluate(x_mid, q=q), "o", color="#2a78d6", ms=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "03_piecewise_interpolation.png")
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
