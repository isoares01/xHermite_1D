"""
Tests added in v1.1.1 (review of the JCAM manuscript). They encode:

* Remark "What the deformation acts through": with the prefactor 1/exp_q(-t^2/2)
  instead of exp_q(t^2/2) every h_n is a polynomial (the deformation disappears);
* exactness of the basis and of M_q near q = 1 (no silent replacement by q = 1);
* Section 4.4 / Proposition 11: centered level-relative variant;
* left-anchored level-relative variant: O(h) for fixed q;
* the L2 counterpart of the saturation constant, |q-1| / (4 sqrt(630)).
"""
import numpy as np
import sympy as sp

from xhermite.assessment import blending, weight
from xhermite.basis import (M1, Q1_SWITCH_TOL, compute_Mq_matrix, compute_Mq_matrix_factorized,
                            h0_q_numeric, q_hermite_basis_numeric)
from xhermite.mathematica_reference import h0 as h0_stable
from xhermite.spline import QHermiteSpline1D
from xhermite.variants import centered_segment, left_anchored_segment, piecewise_variant


def test_remark_weight_only_polynomial_prefactor():
    """With prefactor 1/exp_q(-t^2/2) every (-1)^n (D_qR)^n[E] / E is a polynomial of exact degree n
    (so the interpolant would be the classical one: the deformation comes only from the prefactor)."""
    t = sp.symbols("t", positive=True)
    for q in (sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(5, 2)):
        theta = 1 - (1 - q) * t ** 2 / 2
        E = theta ** (1 / (1 - q))
        f = E
        for n in range(4):
            ratio = sp.simplify(sp.powsimp(sp.expand_power_base(f / E, force=True), force=True))
            assert sp.Poly(sp.expand(ratio), t).degree() == n, (q, n, ratio)
            f = theta * sp.diff(f, t)


def test_basis_exact_near_q1():
    t = np.linspace(0.0, 1.0, 41)
    for q in (0.981, 0.995, 1.005, 1.019, 1.021):
        assert np.allclose(h0_q_numeric(t, q), h0_stable(t, q), rtol=1e-12, atol=1e-12)
    assert np.allclose(h0_q_numeric(t, 1.0), 1.0)
    # the classical basis is recovered exactly at q = 1 only
    h = q_hermite_basis_numeric(t, 1.0)
    assert np.allclose(h[3], t ** 3 - 3 * t)
    assert not np.allclose(h0_q_numeric(np.array([1.0]), 1.019), 1.0, atol=1e-3)


def test_Mq_factorized_matches_closed_form_and_is_continuous():
    for q in (-0.7, 0.4, 0.9, 1.1, 1.6, 2.4, 2.8):
        if abs(q - 1.0) < Q1_SWITCH_TOL:
            continue
        A = compute_Mq_matrix(q)
        B = compute_Mq_matrix_factorized(q)
        assert np.allclose(A, B, rtol=1e-10, atol=1e-10)
    assert np.allclose(compute_Mq_matrix_factorized(1.0), M1)
    # no jump across the switch
    lo = compute_Mq_matrix(1.0 + Q1_SWITCH_TOL - 1e-9)
    hi = compute_Mq_matrix(1.0 + Q1_SWITCH_TOL + 1e-9)
    assert np.allclose(lo, hi, atol=1e-6)


def test_Mq_near_q1_satisfies_hermite_conditions():
    R = np.array([0.3, 1.7, -0.5, 2.1])
    for q in (0.985, 0.999, 1.004, 1.015):
        c = R @ compute_Mq_matrix(q)
        h = q_hermite_basis_numeric(np.array([0.0, 1.0]), q)
        s0 = sum(c[k] * h[k][0] for k in range(4))
        s1 = sum(c[k] * h[k][1] for k in range(4))
        assert np.isclose(s0, R[0], atol=1e-10) and np.isclose(s1, R[1], atol=1e-10)


def test_second_derivative_analytic_matches_finite_differences():
    x = np.linspace(0, 1, 4)
    sp_ = QHermiteSpline1D(x, np.sin(3 * x), 3 * np.cos(3 * x))
    t = np.array([0.1, 0.5, 0.9])
    d2 = sp_.second_derivative(t, 1, 0.7, x[2] - x[1])
    eps = 1e-4
    f = lambda tt: sp_.evaluate(x[1] + tt * (x[2] - x[1]), 0.7)
    fd = (f(t + eps) - 2 * f(t) + f(t - eps)) / eps ** 2 / (x[2] - x[1]) ** 2
    assert np.allclose(d2, fd, rtol=1e-4, atol=1e-4)


def test_centered_variant_properties():
    t = np.linspace(0, 1, 101)
    for q in (-0.7, 0.5, 1.0, 2.6):
        R = (0.3, -0.9, 1.1, 0.4)
        Rr = (R[1], R[0], -R[3], -R[2])
        assert np.allclose(centered_segment(1 - t, q, *Rr), centered_segment(t, q, *R), atol=1e-12)
        assert np.allclose(centered_segment(t, q, 2.0, 2.0, 0.0, 0.0), 2.0, atol=1e-12)
        shifted = centered_segment(t, q, R[0] + 5, R[1] + 5, R[2], R[3]) - 5
        assert np.allclose(shifted, centered_segment(t, q, *R), atol=1e-12)
        ends = centered_segment(np.array([0.0, 1.0]), q, *R)
        assert np.allclose(ends, R[:2], atol=1e-12)
    # q = 1: classical Hermite
    R = (0.3, -0.9, 1.1, 0.4)
    classical = (R[0] * (2 * t ** 3 - 3 * t ** 2 + 1) + R[1] * (-2 * t ** 3 + 3 * t ** 2)
                 + R[2] * (t ** 3 - 2 * t ** 2 + t) + R[3] * (t ** 3 - t ** 2))
    assert np.allclose(centered_segment(t, 1.0, *R), classical, atol=1e-12)


def test_left_anchored_variant_constants_and_first_order():
    t = np.linspace(0, 1, 51)
    assert np.allclose(left_anchored_segment(t, 2.6, 3.0, 3.0, 0.0, 0.0), 3.0)
    xd = (np.arange(2000) + 0.5) / 2000
    for fn in (left_anchored_segment, centered_segment):
        for q in (0.5, 1.5, 2.6):
            errs = []
            for n in (16, 32, 64):
                x = np.linspace(0, 1, n + 1)
                errs.append(np.max(np.abs(piecewise_variant(fn, x, np.exp(x), np.exp(x), q, xd) - np.exp(xd))))
            eoc = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
            assert np.all(np.abs(eoc - 1.0) < 0.1), (fn.__name__, q, eoc)


def test_L2_defect_constant():
    t = np.linspace(0, 1, 20001)
    for eps in (1e-3, -1e-3):
        phi = blending(t, 1.0 + eps)
        d = 1.0 - phi[0] - phi[1]
        l2 = np.sqrt(np.trapezoid(d * d, t))
        assert np.isclose(l2 / abs(eps), 1.0 / (4.0 * np.sqrt(630.0)), rtol=2e-3)


def test_weight_equals_q_exponential_of_q_sum():
    """w_q = exp_q(t^2/2) exp_q(-t^2/2) = exp_q(-(1-q)t^4/4) (pseudo-additivity of the q-sum, Section 2.1)."""
    def expq(x, q):
        return np.maximum(1 + (1 - q) * x, 0.0) ** (1.0 / (1 - q))
    t = np.linspace(0.0, 1.0, 21)
    for q in (-0.7, 0.4, 1.6, 2.8):
        w = weight(t, q, 0)[0]
        assert np.allclose(w, expq(t ** 2 / 2, q) * expq(-t ** 2 / 2, q), rtol=1e-12)
        assert np.allclose(w, expq(-(1 - q) * t ** 4 / 4, q), rtol=1e-12)
    # duality 1/exp_q(-x) = exp_{2-q}(x)
    x = np.linspace(-0.5, 0.5, 11)
    for q in (0.4, 1.6):
        assert np.allclose(1.0 / expq(-x, q), expq(x, 2 - q), rtol=1e-12)


def test_blending_change_below_two_percent_for_q_05_15():
    """Used in the Conclusions: for 0.5 <= q <= 1.5 each blending function moves by < 2%."""
    t = np.linspace(0, 1, 801)
    psi = np.array([2 * t ** 3 - 3 * t ** 2 + 1, -2 * t ** 3 + 3 * t ** 2, t ** 3 - 2 * t ** 2 + t, t ** 3 - t ** 2])
    dev = max(np.max(np.abs(blending(t, q) - psi)) for q in np.linspace(0.5, 1.5, 41))
    assert dev < 0.02


def test_deprecated_additivity_alias():
    from xhermite import additivity, constant_reproduction
    assert additivity is constant_reproduction
