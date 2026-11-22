"""
Oracle-free heuristics for selecting the deformation parameter q,
implementing the two data-driven/oracle-free criteria that are evaluated in the
article (Section 5.1, "Selection of q"; results in Sections 5.3-5.4, methods
M2 and M3, together with leave-one-out cross-validation, ``assessment.loocv_q``):

  1. Strain-energy minimization (minimal bending energy):
         q* = argmin_q  sum_i (1/dx_i^3) int_0^1 [d^2 s_i^(q)/dt^2]^2 dt
     NOTE: this criterion is degenerate. On each segment the classical cubic
     Hermite interpolant minimizes int (s'')^2 among ALL C^2 functions with the
     same end values and end derivatives, and every x-Hermite segment satisfies
     those same four conditions; hence the minimum is always attained at q = 1
     (checked numerically in tests/test_benchmark.py). It is kept here only to
     document that fact.

  2. C2-jump minimization at interior knots (C2-jump criterion):
         q* = argmin_q  sum_i [ s''_{i-1}(1) - s''_i(0) ]^2

Both criteria require only the prescribed Hermite restrictions R_i --
NOT the target function -- unlike the oracle search of
``weierstrass.oracle_q_search``. Their performance against the oracle
q_Emin of the lattice-sampling tables (Appendix C) and of the randomized benchmark
is reported in Sections 5.3-5.4 of the article: neither criterion beats q = 1.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

from .basis import ADMISSIBLE_Q_MAX, ADMISSIBLE_Q_MIN
from .spline import QHermiteSpline1D

_N_QUAD_POINTS = 41  # fixed Simpson's-rule grid; the integrand (from a
                     # finite-difference 2nd derivative) is smooth enough
                     # that adaptive quadrature is unnecessary overhead
                     # and can otherwise trigger spurious roundoff warnings.


def strain_energy(spline: QHermiteSpline1D, q: float) -> float:
    """Total bending energy of the composite interpolant for a given q."""
    total = 0.0
    t_grid = np.linspace(0.0, 1.0, _N_QUAD_POINTS)
    for i in range(spline.num_segments):
        dx = spline.x[i + 1] - spline.x[i]
        d2s = spline.second_derivative(t_grid, i, q, dx)
        total += simpson(d2s ** 2, x=t_grid) / dx ** 3
    return total


def c2_jump_energy(spline: QHermiteSpline1D, q: float) -> float:
    """Sum of squared second-derivative jumps at interior knots."""
    total = 0.0
    for i in range(1, spline.num_segments):
        dx_prev = spline.x[i] - spline.x[i - 1]
        dx_cur = spline.x[i + 1] - spline.x[i]
        left = spline.second_derivative(np.array([1.0]), i - 1, q, dx_prev)[0]
        right = spline.second_derivative(np.array([0.0]), i, q, dx_cur)[0]
        total += (left - right) ** 2
    return total


def _safe_bounded_search(objective, bounds, n_probe: int = 60) -> Tuple[Optional[float], float]:
    """
    Brent-style bounded 1-D search over the admissible q range, skipping
    points too close to the singularities q in {-1, 1, 2, 3} (Section 3).
    A coarse probe grid first avoids Brent converging into a singularity.
    Only the true singularities q = -1 and q = 3 are avoided: q = 1 and q = 2
    are removable for the interpolant (see mathematica_reference.xhermite_segment).
    """
    lo, hi = bounds
    probe_q = np.linspace(lo, hi, n_probe)
    probe_q = probe_q[(probe_q > -1.0 + 0.02) & (probe_q < 3.0 - 0.02)]

    best_q, best_val = None, np.inf
    for q in probe_q:
        try:
            val = objective(float(q))
        except Exception:
            continue
        if val < best_val:
            best_val, best_q = val, float(q)

    if best_q is None:
        return None, np.inf

    # Local refinement with a bounded Brent search around the coarse minimum.
    step = (hi - lo) / n_probe
    lo_r, hi_r = max(lo, best_q - 2 * step), min(hi, best_q + 2 * step)
    try:
        res = minimize_scalar(objective, bounds=(lo_r, hi_r), method="bounded")
        if res.success and res.fun < best_val:
            return float(res.x), float(res.fun)
    except Exception:
        pass
    return best_q, best_val


def select_q_strain_energy(
    spline: QHermiteSpline1D,
    q_bounds: Tuple[float, float] = (ADMISSIBLE_Q_MIN + 0.02, ADMISSIBLE_Q_MAX - 0.02),
) -> Tuple[Optional[float], float]:
    """Select q by minimizing strain_energy(spline, q). Returns (q*, energy)."""
    return _safe_bounded_search(lambda q: strain_energy(spline, q), q_bounds)


def select_q_c2_jump(
    spline: QHermiteSpline1D,
    q_bounds: Tuple[float, float] = (ADMISSIBLE_Q_MIN + 0.02, ADMISSIBLE_Q_MAX - 0.02),
) -> Tuple[Optional[float], float]:
    """Select q by minimizing c2_jump_energy(spline, q). Returns (q*, jump)."""
    if spline.num_segments < 2:
        raise ValueError("C2-jump criterion requires at least 2 segments (1 interior knot).")
    return _safe_bounded_search(lambda q: c2_jump_energy(spline, q), q_bounds)
