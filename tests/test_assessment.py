"""Tests for the analysis added in the 2026 (JCAM) revision: xhermite.assessment."""
import numpy as np
import pytest

from xhermite import assessment as A
from xhermite.mathematica_reference import xhermite_segment

QS = [-0.9, -0.5, 0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5, 2.95]


@pytest.mark.parametrize("q", QS)
def test_blending_matches_factorised_segment(q):
    t = np.linspace(0, 1, 51)
    s = A.segment(0.3, 1.2, 0.7, -0.4, t, q)
    assert np.allclose(s, xhermite_segment(t, q, 0.3, 1.2, 0.7, -0.4), atol=1e-13)


@pytest.mark.parametrize("q", QS)
def test_hermite_conditions_and_derivatives(q):
    e = np.array([0.0, 1.0])
    assert np.allclose(A.blending(e, q), [[1, 0], [0, 1], [0, 0], [0, 0]], atol=1e-13)
    assert np.allclose(A.blending(e, q, 1), [[0, 0], [0, 0], [1, 0], [0, 1]], atol=1e-12)
    t, h = np.linspace(0.1, 0.9, 9), 1e-5
    fd1 = (A.blending(t + h, q) - A.blending(t - h, q)) / (2 * h)
    fd2 = (A.blending(t + h, q) - 2 * A.blending(t, q) + A.blending(t - h, q)) / h**2
    assert np.allclose(A.blending(t, q, 1), fd1, atol=1e-6)
    assert np.allclose(A.blending(t, q, 2), fd2, atol=1e-3)


def test_weight_is_smooth_through_q1():
    t = np.linspace(0, 1, 11)
    for s in (1e-3, 1e-6, 1e-9):
        assert np.allclose(A.weight(t, 1 + s)[0], 1 + s * t**4 / 4, atol=10 * s**2)


def test_gamma_first_order_constant():
    """gamma_q = |q-1|/64 + O((q-1)^2)  (Lemma: constant-reproduction defect)."""
    for s in (1e-2, -1e-2, 1e-3, -1e-3):
        # relative second-order correction is O(|s|); observed ~0.98|s|
        assert abs(A.gamma_q(1 + s) * 64 / abs(s) - 1) < 2 * abs(s)
    assert A.gamma_q(1.0) < 1e-14


def test_defect_profile_first_order():
    t = np.linspace(0, 1, 101)
    s = 1e-4
    phi = A.blending(t, 1 + s)
    assert np.allclose((1 - phi[0] - phi[1]) / s, -t**2 * (1 - t) ** 2 / 4, atol=1e-3)


def test_reversal_symmetry_only_at_q1():
    assert A.reversal_defect(1.0) < 1e-14
    for q in (0.5, 1.5, 2.6):
        assert A.reversal_defect(q) > 1e-3


def test_translation_defect_equals_sigma():
    t = np.linspace(0, 1, 21)
    q, c = 2.6, 5.0
    base = A.segment(0.3, 1.2, 0.7, -0.4, t, q)
    shifted = A.segment(0.3 + c, 1.2 + c, 0.7, -0.4, t, q)
    phi = A.blending(t, q)
    assert np.allclose(shifted - base, c * (phi[0] + phi[1]))


def test_lipschitz_bound_holds():
    L = A.lipschitz_constant(0.0, 2.0)
    for q in np.linspace(0.0, 2.0, 37):
        assert np.max(A.deviation(q)) <= L * abs(q - 1) + 1e-12


def test_loocv_selects_one_for_cubic_data():
    x = np.linspace(0, 1, 9)
    f = lambda z: 1 + z - 2 * z**2 + 3 * z**3
    df = lambda z: 1 - 4 * z + 9 * z**2
    q, score = A.loocv_q(x, f(x), "exact", Dfun=df)
    assert q == 1.0 and score.min() < 1e-24


def test_plateau_matches_theorem():
    """Fixed q != 1, exact tangents: ||f - S_q|| -> gamma_q ||f||_inf (Theorem)."""
    f, df = np.exp, np.exp
    x = np.linspace(0, 1, 257)
    xd = (np.arange(4000) + 0.5) / 4000
    for q in (0.5, 1.5):
        e = np.max(np.abs(A.piecewise(x, f(x), df(x), q, xd) - f(xd)))
        assert abs(e / (A.gamma_q(q) * np.e) - 1) < 0.01


def test_convergence_order_with_q_of_h():
    f, df = np.sin, np.cos
    xd = (np.arange(3000) + 0.5) / 3000
    errs = []
    for n in (64, 128, 256):
        x = np.linspace(0, 1, n + 1)
        q = 1 + 0.5 * (n / 16.0) ** -2
        errs.append(np.max(np.abs(A.piecewise(x, f(x), df(x), q, xd) - f(xd))))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(rates - 2) < 0.15)


def test_defect_second_order():
    """1 - sigma_q = -(e/4) g - (e^2/32) g (t^4-2t^3-3t^2-4t-5) + O(e^3), g = t^2(1-t)^2."""
    t = np.linspace(0, 1, 101)
    g = t**2 * (1 - t) ** 2
    for e in (2e-3, -2e-3):
        phi = A.blending(t, 1 + e)
        approx = -(e / 4) * g - (e**2 / 32) * g * (t**4 - 2 * t**3 - 3 * t**2 - 4 * t - 5)
        assert np.max(np.abs((1 - phi[0] - phi[1]) - approx)) < 50 * abs(e) ** 3


def test_sign_of_defect_on_grid():
    t = np.linspace(0.01, 0.99, 99)
    for q in A.Q_GRID:
        if abs(q - 1) > 1e-12:
            assert np.all(np.sign(1 - A.blending(t, q)[:2].sum(0)) == np.sign(1 - q))
