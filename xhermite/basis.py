"""
Third-degree x-Hermite basis and transformation matrix.

Implements Eq. (basis) (the functions h_0..h_3) and Appendix A (the matrix M_q)
of the article, with the basis truncated with the "positive part" convention of
Eq. (q_exponential): outside the interval I_q = (-t_max(q), t_max(q)), where
4 - t^4 (q-1)^2 > 0, the functions are not defined in the real domain. For q < 1
the positive-part convention sets them to zero; for q > 1 they diverge at the
endpoints of I_q. This code sets them to zero in both cases (a numerical
convention, never used inside the admissible window; see ``boundary.py`` and
Appendix B of the article, which keeps the complex continuation instead).

Standard window and domain of validity
--------------------------------------
All segments are parametrised on the standard Hermite/Bezier/B-spline window
t in [0, 1]. The basis is real on I_q, and [0, 1] is contained in I_q exactly
when -1 < q < 3 (Section 3 of the article). That range is therefore a
consequence of the choice of window, not an intrinsic limit of the Rodrigues
construction (a shorter window [0, T], T < t_max(q), is possible for other q;
see Section 4.4 of the article).

Numerical safety margin
-----------------------
In the article's experiments q is restricted to [-0.8, 2.9], i.e. it stays
inside (-1, 3) with a margin: near q = -1 and q = 3 the weight w_q(1) tends
to 0 or infinity, the constants L_Q and gamma_q blow up (Fig. gamma) and the
interpolant becomes very sensitive to q. A margin therefore improves stability.

Behaviour near q = 1 (change in v1.1.1)
---------------------------------------
* ``h0_q_numeric`` / ``q_hermite_basis_numeric`` evaluate the weight through
  the analytic representation of Lemma 1,
  w_q(t) = exp((q-1) t^4/4 * G((q-1)^2 t^4/4)), G(x) = -ln(1-x)/x, which is
  valid through q = 1 and never overflows. (Versions <= 1.1.0 replaced the whole
  interval |q-1| < 0.02 by the classical basis, which changes h_0(1) by up to
  5e-3.)
* ``compute_Mq_matrix``: the closed-form entries of Appendix A contain factors
  such as 2^(1/(q-1)) that overflow in double precision for |q-1| of the order
  of 1e-3 and lose accuracy to cancellation well before that. For
  |q-1| < ``Q1_SWITCH_TOL`` the matrix is therefore computed from the
  factorized form of Theorem 3, M_q = T_q M_H B_q (see
  ``compute_Mq_matrix_factorized``), which is exact and tends to M_1.

IMPORTANT CORRECTION w.r.t. the matrix at q = 1
-----------------------------------------------
The classical Hermite matrix M_H of Eq. (M1_matrix) is designed to combine with
the *canonical monomial* basis B_can(t) = (1, t, t^2, t^3)^T, via
S(t) = R . M_H . B_can(t). It does **not** combine correctly with the classical
Rodrigues basis H_1^3(t) = (1, t, t^2-1, t^3-3t)^T (this was a bug in an
earlier version of the manuscript). The correct matrix is M_1 = M_H @ A, where
B_can(t) = A . H_1^3(t):

    A = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 3, 0, 1]]

    M_1 = [[-2,  6, -3,  2],
           [ 3, -6,  3, -2],
           [-2,  4, -2,  1],
           [-1,  3, -1,  1]].

This equals lim_{q->1} M_q (``tests/test_core.py::test_M1_matches_limit_of_Mq``,
``symbolic.py::verify_M1_is_limit_of_Mq``).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Admissible range of q for the standard window t in [0, 1] (Section 3 of the article)
# ---------------------------------------------------------------------------
ADMISSIBLE_Q_MIN = -1.0
ADMISSIBLE_Q_MAX = 3.0
ADMISSIBLE_Q_EXCLUDED = (1.0, 2.0)  # q = 1 handled separately; q = 2 is a
                                    # degeneracy of the basis (Theorem 3(i))


def is_admissible(q: float, eps: float = 1e-6) -> bool:
    """True if q lies in (-1, 3) and is not (numerically) 1 or 2."""
    if not (ADMISSIBLE_Q_MIN < q < ADMISSIBLE_Q_MAX):
        return False
    return all(abs(q - qx) > eps for qx in ADMISSIBLE_Q_EXCLUDED)


#: Half-width of the neighbourhood of q = 1 in which M_q is computed from the
#: factorized form (exact) instead of the Appendix A closed form. The closed-form
#: terms scale like exp(C/(q-1)) with C up to ~7; float64 overflows for
#: |q-1| below roughly 7/1024 ~ 0.007, and cancellation degrades accuracy
#: earlier, so a comfortable margin is used.
Q1_SWITCH_TOL = 0.02


# ---------------------------------------------------------------------------
# Classical (q = 1) reference matrices
# ---------------------------------------------------------------------------

#: Classical Hermite matrix M_H (Eq. (M1_matrix)). Combines with the canonical monomial
#: basis B_can(t) = (1, t, t^2, t^3)^T, NOT with H_1^3(t). Kept here mostly
#: for reference / cross-checking; end users should use ``M1`` below.
MH_canonical = np.array([
    [1.0, 0.0, -3.0, 2.0],
    [0.0, 0.0, 3.0, -2.0],
    [0.0, 1.0, -2.0, 1.0],
    [0.0, 0.0, -1.0, 1.0],
])

#: Change of basis matrix A such that B_can(t) = A . H_1^3(t).
_A_CHANGE_OF_BASIS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 3.0, 0.0, 1.0],
])

#: Correct q = 1 coefficient matrix, to be combined with
#: H_1^3(t) = (1, t, t^2-1, t^3-3t)^T. See module docstring.
M1 = MH_canonical @ _A_CHANGE_OF_BASIS


# ---------------------------------------------------------------------------
# Third-degree x-Hermite basis functions (Eq. (basis))
# ---------------------------------------------------------------------------

def _G(x: np.ndarray) -> np.ndarray:
    """G(x) = -ln(1-x)/x for x < 1, with G(0) = 1 (series for tiny |x|)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 0.5, x)
    series = 1.0 + x / 2.0 + x * x / 3.0
    return np.where(small, series, -np.log1p(-xs) / xs)


def h0_q_numeric(t: np.ndarray, q: float) -> np.ndarray:
    """
    Weight w_q(t) = h_{0(q)}^3(t) = [1 - (q-1)^2 t^4 / 4]^{1/(1-q)} (Eq. (basis)),
    evaluated through the analytic representation of Lemma 1,
    exp((q-1) t^4/4 * G((q-1)^2 t^4/4)), which is valid through q = 1 (where it
    equals 1) and does not overflow. Outside I_q, i.e. wherever
    (q-1)^2 t^4 / 4 >= 1, the weight is not defined (it vanishes for q < 1 and
    diverges for q > 1); the code returns 0 there as a numerical convention.
    """
    t = np.asarray(t, dtype=float)
    e = q - 1.0
    x = e * e * t ** 4 / 4.0
    inside = x < 1.0
    xs = np.where(inside, x, 0.0)
    val = np.exp(e * t ** 4 / 4.0 * _G(xs))
    return np.where(inside, val, 0.0)


def q_hermite_basis_numeric(t: np.ndarray, q: float):
    """
    Return (h0, h1, h2, h3), the four third-degree x-Hermite basis functions of
    Eq. (basis), evaluated at t for a given q. For q = 1 this is exactly the
    classical basis {1, t, t^2 - 1, t^3 - 3t} (k1 = k2 = 1, k3 = 3).
    """
    t = np.asarray(t, dtype=float)
    h0 = h0_q_numeric(t, q)
    k1 = (3.0 - q) / 2.0
    k2 = (q - 3.0) * (q - 2.0) / 2.0
    k3 = 4.0 - q
    h1 = t * h0
    h2 = (k1 * t ** 2 - 1.0) * h0
    h3 = (k2 * t ** 3 - k3 * t) * h0
    return h0, h1, h2, h3


# ---------------------------------------------------------------------------
# Transformation matrix M_q (Appendix A and Theorem 3)
# ---------------------------------------------------------------------------

def compute_Mq_matrix_factorized(q: float) -> np.ndarray:
    """
    M_q obtained from the factorized form of Theorem 3, valid for q != 2, 3
    and numerically stable through q = 1:

        M_q = T_q  M_H  B_q,

    where T_q maps R = [P0, Pf, V0, Vf] to the data of the classical cubic p
    (Eq. (modified_restrictions)), M_H is the classical Hermite matrix and B_q
    expresses the monomials in the Rodrigues polynomials
    [1, t, k1 t^2 - 1, k2 t^3 - k3 t]. For q = 1 it returns M_1 = M_H A.
    """
    from .assessment import weight  # analytic w, w' (stable through q = 1)

    a, b = (float(v) for v in weight(1.0, q, 1))
    k1 = (3.0 - q) / 2.0
    k2 = (q - 3.0) * (q - 2.0) / 2.0
    k3 = 4.0 - q
    if k1 == 0.0 or k2 == 0.0:
        raise ValueError(f"q={q}: the Rodrigues basis degenerates (q in {{2, 3}}).")
    T = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0 / a, 0.0, -b / a ** 2],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0 / a]])
    B = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [1.0 / k1, 0.0, 1.0 / k1, 0.0],
                  [0.0, k3 / k2, 0.0, 1.0 / k2]])
    return T @ MH_canonical @ B


def compute_Mq_matrix(q: float) -> np.ndarray:
    """
    Compute the 4x4 transformation matrix M_q mapping the Hermite restriction
    vector R_i = [P_i, P_{i+1}, V_i, V_{i+1}] to the coefficients
    c = R_i @ M_q of the local x-Hermite segment (Eq. (complete_q_hermite_segment)),
    following Appendix A of the article.

    For |q - 1| < ``Q1_SWITCH_TOL`` the Appendix A closed form (which contains
    2**(1/(q-1)) and similar factors, overflowing near q = 1) is replaced by
    the algebraically identical factorized computation
    (``compute_Mq_matrix_factorized``); at q = 1 exactly it returns M_1.
    """
    if abs(q - 1.0) < 1e-12:
        return M1.copy()
    if abs(q - 1.0) < Q1_SWITCH_TOL:
        return compute_Mq_matrix_factorized(q)

    if not is_admissible(q, eps=1e-3):
        raise ValueError(
            f"q={q} is outside/near the admissible domain "
            f"({ADMISSIBLE_Q_MIN}, {ADMISSIBLE_Q_MAX}) \\ {{1, 2}}; "
            "M_q is singular or numerically ill-conditioned there (Section 3 of the article)."
        )

    return compute_Mq_matrix_closed(q)


def compute_Mq_matrix_closed(q: float) -> np.ndarray:
    """
    Appendix A closed form of M_q, evaluated literally in double precision with
    no admissibility check and no switch to the factorized form. It is exposed
    for the stability study of ``singularity.py`` (it overflows or cancels near
    q = 1, has poles at q = 2 and q = 3 and is badly scaled near q = -1); for
    ordinary use call ``compute_Mq_matrix``.
    """
    m00 = (3.0 + q) / (q - 3.0)
    m01 = (16.0 - 4.0 * q) / ((q - 3.0) * (q - 2.0))
    m02 = 6.0 / (q - 3.0)
    m03 = 4.0 / ((q - 3.0) * (q - 2.0))

    m10 = (2.0 ** ((q - 4.0) / (q - 1.0))) * (q + 1.0) * \
          ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0) - 2.0)) * \
          (5.0 * 2.0 ** (1.0 / (q - 1.0)) + 5.0 * 2.0 ** (q / (q - 1.0)) * q
           - 3.0 * 2.0 ** (1.0 / (q - 1.0)) * q ** 2)
    m11 = (2.0 ** (-7.0 / (q - 1.0))) * (q + 1.0) * \
          ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0) - 2.0)) / (q - 2.0) * \
          (2.0 ** (4.0 + 5.0 / (q - 1.0)) + 15.0 * 2.0 ** (2.0 + 5.0 / (q - 1.0)) * q
           - 2.0 ** (5.0 * q / (q - 1.0)) * q ** 2 + 2.0 ** (2.0 + 5.0 / (q - 1.0)) * q ** 3)
    m12 = (2.0 ** ((q - 3.0) / (q - 1.0))) * (q + 1.0) * \
          ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0) - 2.0)) * (5.0 + 10.0 * q - 3.0 * q ** 2)
    m13 = (2.0 ** ((2.0 * q - 4.0) / (q - 1.0))) * (q + 1.0) * \
          ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0) - 2.0)) * \
          (1.0 + 4.0 * q - q ** 2) / (q - 2.0)

    m20 = 4.0 / (q - 3.0)
    m21 = (14.0 - 7.0 * q + q ** 2) / ((q - 3.0) * (q - 2.0))
    m22 = 4.0 / (q - 3.0)
    m23 = 2.0 / ((q - 3.0) * (q - 2.0))

    m30 = (2.0 ** ((q - 3.0) / (q - 1.0))) * ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0))) / (q - 3.0)
    m31 = (2.0 ** ((q - 3.0) / (q - 1.0))) * (q - 4.0) * \
          ((3.0 + 2.0 * q - q ** 2) ** (q / (q - 1.0))) / \
          (((q - 3.0) ** 2) * (q - 2.0) * (q + 1.0))
    m32 = (2.0 ** ((q - 3.0) / (q - 1.0))) * ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0))) / (q - 3.0)
    m33 = (2.0 ** ((q - 3.0) / (q - 1.0))) * ((3.0 + 2.0 * q - q ** 2) ** (1.0 / (q - 1.0))) / ((q - 3.0) * (q - 2.0))

    return np.array([
        [m00, m01, m02, m03],
        [m10, m11, m12, m13],
        [m20, m21, m22, m23],
        [m30, m31, m32, m33],
    ])
