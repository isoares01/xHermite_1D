"""
Piecewise x-Hermite spline (Section 3.3 of the article).

Supports a single global q, a different q per segment (figuras/fig5b_per_segment.pdf), and either prescribed derivatives or derivatives estimated
from the data.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .basis import compute_Mq_matrix
from .mathematica_reference import xhermite_segment


class QHermiteSpline1D:
    """
    Piecewise x-Hermite interpolant through (x_i, y_i) with prescribed
    (or supplied) derivatives dy_i, following Section 3.3 (piecewise interpolant) and Eq. (complete_q_hermite_segment)
    of the article.

    Parameters
    ----------
    x_points, y_points : array_like
        Interpolation nodes (Section 3.3) and prescribed values P_i.
    dy_points : array_like, optional
        Prescribed derivatives V_i (w.r.t. x). If omitted, defaults to
        zero at every node, which is exactly the setting of the article's
        Weierstrass benchmark (zero tangents; ``weierstrass.py`` explains why
        the term-by-term derivative of the truncated series is not used).
    """

    def __init__(
        self,
        x_points: Sequence[float],
        y_points: Sequence[float],
        dy_points: Optional[Sequence[float]] = None,
    ):
        self.x = np.asarray(x_points, dtype=float)
        self.y = np.asarray(y_points, dtype=float)
        if self.x.ndim != 1 or len(self.x) < 2:
            raise ValueError("x_points must be a 1-D array with at least 2 nodes.")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x_points must be strictly increasing.")
        if self.y.shape != self.x.shape:
            raise ValueError("y_points must have the same length as x_points.")
        self.num_segments = len(self.x) - 1

        self.dy = (
            np.zeros_like(self.y) if dy_points is None else np.asarray(dy_points, dtype=float)
        )
        if self.dy.shape != self.y.shape:
            raise ValueError("dy_points must have the same length as y_points.")
        self._Mq_cache = {}

    def _Mq(self, q: float) -> np.ndarray:
        key = round(float(q), 12)
        if key not in self._Mq_cache:
            self._Mq_cache[key] = compute_Mq_matrix(q)
        return self._Mq_cache[key]

    def segment_coefficients(self, i: int, q: float) -> np.ndarray:
        """Return c_i = R_i @ M_q (Eq. (complete_q_hermite_segment)) for segment i and parameter q."""
        x0, x1 = self.x[i], self.x[i + 1]
        dx = x1 - x0
        R_i = np.array([self.y[i], self.y[i + 1], self.dy[i] * dx, self.dy[i + 1] * dx])
        return R_i @ self._Mq(q)

    def evaluate(self, x_eval: np.ndarray, q: Union[float, Sequence[float]]) -> np.ndarray:
        """
        Evaluate the piecewise x-Hermite interpolant at x_eval.

        Parameters
        ----------
        x_eval : array_like
            Points at which to evaluate the interpolant.
        q : float or sequence of floats
            A single global deformation parameter, or one value per
            segment (length ``self.num_segments``), reproducing the
            per-interval q selection illustrated in figuras/fig5b_per_segment.pdf
            (Section 5.5).
        """
        x_eval = np.asarray(x_eval, dtype=float)
        y_eval = np.full_like(x_eval, np.nan)

        if np.isscalar(q):
            q_per_segment = [float(q)] * self.num_segments
        else:
            q_per_segment = list(q)
            if len(q_per_segment) != self.num_segments:
                raise ValueError(
                    f"Expected {self.num_segments} values of q (one per segment), "
                    f"got {len(q_per_segment)}."
                )

        for i in range(self.num_segments):
            x0, x1 = self.x[i], self.x[i + 1]
            dx = x1 - x0
            last = i == self.num_segments - 1
            mask = (x_eval >= x0) & ((x_eval <= x1) if last else (x_eval < x1))
            if not np.any(mask):
                continue

            t = (x_eval[mask] - x0) / dx
            # Numerically stable, algebraically identical form of R_i M_q H_q^3(t):
            # s = h0(t) * cubic Hermite with modified data. Valid for every q in
            # (-1, 3), including q = 1 and q = 2 (removable for the interpolant).
            y_eval[mask] = xhermite_segment(t, q_per_segment[i], self.y[i], self.y[i + 1],
                                            self.dy[i] * dx, self.dy[i + 1] * dx)

        return y_eval

    def second_derivative(self, t: np.ndarray, i: int, q: float, dx: float) -> np.ndarray:
        """
        d^2 s_i^(q)/dx^2 at local parameter t in [0,1] for segment i, using the
        closed-form second derivatives of the blending functions
        (``assessment.blending(..., d=2)``) and the chain rule
        d^2/dx^2 = (1/dx^2) d^2/dt^2. Used by the oracle-free heuristics of
        ``heuristics.py`` (Section 5.1, criteria for selecting q). (Versions
        <= 1.1.0 used a finite-difference approximation with step 1e-5.)
        """
        from .assessment import blending

        t = np.atleast_1d(np.asarray(t, dtype=float))
        R = np.array([self.y[i], self.y[i + 1], self.dy[i] * dx, self.dy[i + 1] * dx])
        d2s_dt2 = R @ blending(t, q, d=2)
        return d2s_dt2 / dx ** 2
