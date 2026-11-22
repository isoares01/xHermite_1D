"""
Reproduces Figure 1 of the article: the third-degree x-Hermite basis
functions for q = -0.8, 0.8, 1.0 (classical), 2.5.

Usage:
    python examples/fig_basis.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from xhermite.basis import q_hermite_basis_numeric

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def plot_basis(ax, q, t_range):
    t = np.linspace(*t_range, 800)
    h0, h1, h2, h3 = q_hermite_basis_numeric(t, q)
    labels = [r"$H_{q,0}^3(t)$", r"$H_{q,1}^3(t)$", r"$H_{q,2}^3(t)$", r"$H_{q,3}^3(t)$"]
    for h, lab in zip((h0, h1, h2, h3), labels):
        ax.plot(t, h, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"q = {q}" if q != 1.0 else "q = 1 (classical)")
    ax.legend(fontsize=7)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    configs = [(-0.8, (-1.1, 1.1)), (0.8, (-4, 4)), (1.0, (-3, 2.2)), (2.5, (-2.2, 2.2))]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (q, t_range) in zip(axes.ravel(), configs):
        plot_basis(ax, q, t_range)

    fig.suptitle("Third-degree x-Hermite basis functions (article, basis figure)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_basis_functions.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
