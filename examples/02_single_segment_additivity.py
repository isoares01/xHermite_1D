"""
Example 02 -- one segment, zero tangents: level amplification (sigma>1) and attenuation (sigma<1)
(article: Lemma 6 "Constants are not reproduced", figuras/fig4_constants_*.pdf).

With V_i = V_{i+1} = 0 the x-Hermite segment is
    s(t) = alpha_q(t) P_i + beta_q(t) P_{i+1},
and alpha_q + beta_q = 1 only for the classical case q = 1.

Usage:
    python examples/02_single_segment_additivity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from xhermite import constant_reproduction, xhermite_segment

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    print("Midpoint weights s(1/2) = alpha*P_i + beta*P_{i+1} (zero tangents)")
    print(f"{'q':>6} {'alpha':>8} {'beta':>8} {'alpha+beta':>11}  regime")
    for q in (-0.6, 0.0, 0.5, 1.0, 1.5, 2.0, 2.6, 2.9):
        a, b, s = constant_reproduction(q)
        regime = "classical" if q == 1 else ("level-amplifying" if s > 1 else "level-attenuating")
        print(f"{q:6.2f} {a:8.4f} {b:8.4f} {s:11.4f}  {regime}")

    # constant data are NOT reproduced for q != 1
    t = np.linspace(0, 1, 201)
    for q in (0.5, 2.5):
        err = np.max(np.abs(xhermite_segment(t, q, 1.0, 1.0) - 1.0))
        print(f"constant data P=1, q={q}: max |s(t)-1| = {err:.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipping the figure)")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    P0, P1 = 0.1, 0.3
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for q, c, ls in ((1.0, "#6b6b6b", "--"), (2.9, "#2a78d6", "-"), (-0.6, "#eb6834", "-")):
        ax.plot(t, xhermite_segment(t, q, P0, P1), ls, color=c, lw=1.6, label=f"q = {q}")
    ax.plot([0, 1], [P0, P1], "ko", ms=5)
    ax.set_xlabel("t")
    ax.set_ylabel("s(t)")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "02_single_segment_additivity.png")
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
