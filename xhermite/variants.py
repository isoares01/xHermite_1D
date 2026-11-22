"""
Level-relative variants of the x-Hermite segment (Section 4.4 of the article).

These are *not* used in the numerical experiments of Section 5; they implement
the constructions discussed in Section 4.4 so that their stated properties can
be checked (``tests/test_variants.py``).

left_anchored_segment
    s = P0 + s^(q)[0, Pf - P0, V0, Vf]. Translation covariant, reproduces
    constants, O(h) for fixed q, NOT reversal symmetric (anchored at the left node).

centered_segment
    s = (P0 + Pf)/2 + u, where u is the unique element of w~_q P^3,
    w~_q(t) = w_q(2t - 1), with u(0) = -(Pf-P0)/2, u(1) = (Pf-P0)/2,
    u'(0) = V0, u'(1) = Vf (Proposition 11). Linear-map and translation
    covariant, reproduces constants, reversal symmetric, O(h) for fixed q,
    classical Hermite at q = 1.
"""
from __future__ import annotations

import numpy as np

from .assessment import weight
from .mathematica_reference import xhermite_segment


def _hermite(t, p0, p1, m0, m1):
    return (p0 * (2 * t ** 3 - 3 * t ** 2 + 1) + p1 * (-2 * t ** 3 + 3 * t ** 2)
            + m0 * (t ** 3 - 2 * t ** 2 + t) + m1 * (t ** 3 - t ** 2))


def left_anchored_segment(t, q: float, P0, P1, V0=0.0, V1=0.0):
    """P0 + s^(q)[0, P1 - P0, V0, V1] on t in [0, 1] (derivatives w.r.t. t)."""
    return P0 + xhermite_segment(t, q, 0.0, P1 - P0, V0, V1)


def centered_segment(t, q: float, P0, P1, V0=0.0, V1=0.0):
    """Centered, level-relative x-Hermite segment on t in [0, 1] (Proposition 11)."""
    if not (-1.0 < q < 3.0):
        raise ValueError("q must lie in (-1, 3)")
    t = np.asarray(t, dtype=float)
    m, d = 0.5 * (P0 + P1), P1 - P0
    w_end, dw_end = (float(v) for v in weight(1.0, q, 1))      # w_q(1), w_q'(1)
    # w~(t) = w_q(2t-1): w~(0) = w~(1) = w_q(1); w~'(1) = 2 w_q'(1) = -w~'(0)
    a, db = w_end, 2.0 * dw_end
    p0 = (-0.5 * d) / a
    m0 = (V0 - (-db) * p0) / a
    p1 = (0.5 * d) / a
    m1 = (V1 - db * p1) / a
    wt = weight(2.0 * t - 1.0, q, 0)[0]
    return m + wt * _hermite(t, p0, p1, m0, m1)


def piecewise_variant(segment_fn, x, P, D, q, xe):
    """Piecewise interpolant built from ``segment_fn`` (global q), D = dS/dx."""
    x, P, D, xe = (np.asarray(v, dtype=float) for v in (x, P, D, xe))
    n = len(x) - 1
    seg = np.clip(np.searchsorted(x, xe, side="right") - 1, 0, n - 1)
    out = np.empty_like(xe)
    for i in range(n):
        m = seg == i
        if m.any():
            h = x[i + 1] - x[i]
            out[m] = segment_fn((xe[m] - x[i]) / h, q, P[i], P[i + 1], h * D[i], h * D[i + 1])
    return out
