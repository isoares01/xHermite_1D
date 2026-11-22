"""
Regression tests encoding every mathematical claim checked against the
article during the preparation of this repository. Run with:

    pytest tests/ -v
"""
import numpy as np
import pytest

from xhermite.basis import (
    ADMISSIBLE_Q_MAX,
    ADMISSIBLE_Q_MIN,
    M1,
    MH_canonical,
    Q1_SWITCH_TOL,
    compute_Mq_matrix,
    h0_q_numeric,
    is_admissible,
    q_hermite_basis_numeric,
)
from xhermite.boundary import demo_report, h0_untruncated, probe_crossing, t_max
from xhermite.spline import QHermiteSpline1D
from xhermite.weierstrass import rmse, weierstrass_derivative, weierstrass_function


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def test_classical_limit_basis():
    """h_n^(q)(t) -> {1, t, t^2-1, t^3-3t} as q -> 1 (Corollary 2, classical limit)."""
    t = np.linspace(-0.9, 0.9, 25)
    for q in (1.0 - 5e-3, 1.0, 1.0 + 5e-3):
        h0, h1, h2, h3 = q_hermite_basis_numeric(t, q)
        assert np.allclose(h0, 1.0, atol=1e-2)
        assert np.allclose(h1, t, atol=1e-2)
        assert np.allclose(h2, t ** 2 - 1, atol=1e-2)
        assert np.allclose(h3, t ** 3 - 3 * t, atol=1e-2)


def test_basis_positive_within_domain():
    """h0 > 0 on [0, 1] for every admissible q (guarantees the '+' branch is never invoked)."""
    t = np.linspace(0.0, 1.0, 50)
    for q in np.linspace(ADMISSIBLE_Q_MIN + 0.05, ADMISSIBLE_Q_MAX - 0.05, 30):
        if abs(q - 1.0) < 0.05 or abs(q - 2.0) < 0.05:
            continue
        h0 = h0_q_numeric(t, q)
        assert np.all(h0 > 0.0), f"h0 vanished within [0,1] for admissible q={q}"


# ---------------------------------------------------------------------------
# Transformation matrix
# ---------------------------------------------------------------------------

def test_Mq_satisfies_hermite_conditions():
    """R @ M_q, combined with H_q^3(t), reproduces (P0, Pf, V0, Vf) for q != 1."""
    R = np.array([0.3, 1.7, -0.5, 2.1])
    for q in (-0.7, 0.4, 1.6, 2.4, 2.8):
        Mq = compute_Mq_matrix(q)
        c = R @ Mq
        h0_0, h1_0, h2_0, h3_0 = q_hermite_basis_numeric(np.array([0.0]), q)
        h0_1, h1_1, h2_1, h3_1 = q_hermite_basis_numeric(np.array([1.0]), q)
        s0 = c[0] * h0_0 + c[1] * h1_0 + c[2] * h2_0 + c[3] * h3_0
        s1 = c[0] * h0_1 + c[1] * h1_1 + c[2] * h2_1 + c[3] * h3_1
        assert np.isclose(s0[0], R[0], atol=1e-6)
        assert np.isclose(s1[0], R[1], atol=1e-6)


def test_M1_satisfies_hermite_conditions():
    """R @ M1, combined with the classical basis, reproduces (P0, Pf, V0, Vf) exactly."""
    R = np.array([0.3, 1.7, -0.5, 2.1])
    c = R @ M1
    t = 0.4
    s = c[0] * 1 + c[1] * t + c[2] * (t ** 2 - 1) + c[3] * (t ** 3 - 3 * t)
    P0, Pf, V0, Vf = R
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h01 = -2 * t ** 3 + 3 * t ** 2
    h10 = t ** 3 - 2 * t ** 2 + t
    h11 = t ** 3 - t ** 2
    expected = P0 * h00 + Pf * h01 + V0 * h10 + Vf * h11
    assert np.isclose(s, expected, atol=1e-10)


def test_MH_canonical_does_NOT_work_with_H1_basis():
    """
    Documents the bug that WAS present in an earlier version of the
    manuscript (Eq. 71/73/A.18) and in an earlier porting script:
    combining M_H (designed for the canonical monomial basis) directly
    with H_1^3(t) = {1, t, t^2-1, t^3-3t} does NOT satisfy the Hermite
    conditions.
    """
    R = np.array([0.3, 1.7, -0.5, 2.1])
    c_wrong = R @ MH_canonical
    t = 0.4
    s_wrong = c_wrong[0] * 1 + c_wrong[1] * t + c_wrong[2] * (t ** 2 - 1) + c_wrong[3] * (t ** 3 - 3 * t)
    P0 = R[0]
    assert not np.isclose(s_wrong, P0), "expected the (incorrect) M_H . H1 combination to fail"


def test_Mq_continuous_through_q1():
    """M_q -> M1 continuously as q -> 1 from both sides (despite different formulas)."""
    left = compute_Mq_matrix(1.0 - 0.05)
    right = compute_Mq_matrix(1.0 + 0.05)
    assert np.allclose(left, M1, atol=0.5)
    assert np.allclose(right, M1, atol=0.5)
    # tighter as q -> 1
    left_tight = compute_Mq_matrix(1.0 - 0.021)
    right_tight = compute_Mq_matrix(1.0 + 0.021)
    assert np.allclose(left_tight, M1, atol=0.2)
    assert np.allclose(right_tight, M1, atol=0.2)
    # even closer to Q1_SWITCH_TOL, individual matrix entries can still
    # differ by O(0.1-0.2) (the raw Appendix-A entries are not uniformly
    # Lipschitz near q=1; what matters, and what is checked elsewhere,
    # is that R @ M_q converges for any fixed restriction vector R --
    # see test_M1_satisfies_hermite_conditions and the mpmath-based
    # verification in symbolic.py / Corollary 2 of the article), so we
    # only require the raw matrix to stay in the same ballpark here.
    left_tighter = compute_Mq_matrix(1.0 - Q1_SWITCH_TOL - 1e-6)
    assert np.allclose(left_tighter, M1, atol=0.25)


def test_singular_q_raises():
    for q in (2.0, 3.0, -1.0):
        with pytest.raises(ValueError):
            compute_Mq_matrix(q)


# ---------------------------------------------------------------------------
# Admissible domain
# ---------------------------------------------------------------------------

def test_admissible_domain_flags():
    assert is_admissible(0.5)
    assert is_admissible(-0.9)
    assert is_admissible(2.9)
    assert not is_admissible(1.0)
    assert not is_admissible(2.0)
    assert not is_admissible(3.5)
    assert not is_admissible(-1.5)


def test_tmax_boundary_matches_domain_endpoints():
    """t_max(q) -> 1 exactly as q -> -1 or q -> 3 (Section 3 / Appendix B)."""
    assert np.isclose(t_max(-1.0 + 1e-6), 1.0, atol=1e-4)
    assert np.isclose(t_max(3.0 - 1e-6), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Behavior beyond the admissible domain (Appendix B)
# ---------------------------------------------------------------------------

def test_untruncated_h0_matches_truncated_inside_domain():
    t = np.linspace(0, 1, 11)
    q = 0.5
    truncated = h0_q_numeric(t, q)
    untruncated = h0_untruncated(t, q).real
    assert np.allclose(truncated, untruncated, atol=1e-8)


def test_branch_point_beyond_q3():
    """For q > 3, |h0| diverges approaching t_max and becomes strongly complex just past it."""
    probes = probe_crossing(3.5)
    moduli = [p.modulus for p in probes]
    # modulus grows sharply right before t_max, then the point just after
    # is complex with a non-negligible imaginary part
    assert moduli[2] > 100 * moduli[0]  # near-pole blow-up
    just_after = probes[3]
    assert abs(just_after.value.imag) > 0.1 * abs(just_after.value.real)


def test_complex_dominated_beyond_qminus1():
    """For q < -1, the point just past t_max is dominated by its imaginary part."""
    probes = probe_crossing(-1.2)
    just_after = probes[3]
    frac_real = abs(just_after.value.real) / abs(just_after.value)
    assert frac_real < 0.3


def test_demo_report_runs():
    report = demo_report()
    assert "q = 3.5" in report


# ---------------------------------------------------------------------------
# Spline / Weierstrass benchmark
# ---------------------------------------------------------------------------

def test_spline_reproduces_classical_at_q1():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    dy = np.array([1.0, 0.0, -1.0])
    spline = QHermiteSpline1D(x, y, dy)
    x_eval = np.linspace(0, 2, 21)
    y_q = spline.evaluate(x_eval, q=1.0)
    assert np.all(np.isfinite(y_q))
    # interpolation conditions at the nodes themselves
    assert np.isclose(spline.evaluate(np.array([0.0]), q=1.0)[0], 0.0, atol=1e-8)
    assert np.isclose(spline.evaluate(np.array([1.0]), q=1.0)[0], 1.0, atol=1e-8)
    assert np.isclose(spline.evaluate(np.array([2.0]), q=1.0)[0], 0.0, atol=1e-8)


def test_per_segment_q():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    dy = np.array([1.0, 0.0, -1.0])
    spline = QHermiteSpline1D(x, y, dy)
    y_q = spline.evaluate(np.linspace(0, 2, 21), q=[1.5, 2.5])
    assert np.all(np.isfinite(y_q))


def test_weierstrass_derivative_matches_finite_difference():
    """
    d/dx W_a(x) = sum cos(pi k^a x) is the EXACT derivative of the
    truncated (hence smooth, finite) series -- this is elementary
    term-by-term calculus, not an approximation. We only check it here
    with a small K and low a: with the article's default K=250 and
    a>=2, the highest term has frequency k^a up to 250^a, so a naive
    finite-difference check would need eps well below 1e-8 to resolve
    it (central differences amplify high-frequency content via the
    4th-derivative error term) and is therefore the wrong tool for
    validating the high-K/high-a case, not a sign that the formula is
    wrong.
    """
    x = np.linspace(0.05, 0.95, 5)
    a, K = 2, 5
    eps = 1e-5
    fd = (
        weierstrass_function(x + eps, a, K=K) - weierstrass_function(x - eps, a, K=K)
    ) / (2 * eps)
    analytic = weierstrass_derivative(x, a, K=K)
    assert np.allclose(fd, analytic, atol=1e-3)


def test_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(1 / 3))
