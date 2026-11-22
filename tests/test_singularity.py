"""Behaviour near the exceptional values q = -1, 1, 2, 3 (Section 3.5 of the article)."""
import numpy as np
import pytest

from xhermite import singularity as sg
from xhermite.assessment import weight
from xhermite.figures import FIG2_X, FIG2_Y


def test_fig2_data_used_for_the_extent_study_is_the_data_of_the_figure():
    for k, (P, V) in enumerate((FIG2_X, FIG2_Y)):
        for i in range(4):
            assert sg.FIG2_NODES[i][k] == (P[i], V[i])


@pytest.mark.parametrize("q", [-0.99999999, -0.9, 0.3, 1.0, 1.00001, 2.0, 2.9, 2.99999999])
def test_weight_is_accurate_up_to_the_ends(q):
    import mpmath as mp
    t = np.linspace(0, 1, 11)
    w, dw = weight(t, q, 1)
    qq = mp.mpf(q)
    for x, a, b in zip(t, w, dw):
        base = 1 - (qq - 1) ** 2 * mp.mpf(x) ** 4 / 4
        ref = base ** (1 / (1 - qq)) if q != 1 else mp.mpf(1)
        dref = (qq - 1) * mp.mpf(x) ** 3 * base ** (qq / (1 - qq)) if q != 1 else mp.mpf(0)
        assert abs(float((a - ref) / ref)) < 5e-15
        assert abs(float(b - dref)) < 5e-15 * max(1.0, abs(float(dref)))


@pytest.mark.parametrize("q", [-0.9999999, -0.5, 0.5, 1.5, 2.0, 2.5, 2.9999999])
def test_product_form_evaluator_is_accurate_everywhere(q):
    t = np.linspace(0, 1, 101)
    assert sg.rel_err(sg.seg_product(q, t=t), sg.mp_curve(q, t=t)) < 1e-13


def test_blow_up_rates_at_the_genuine_singularities():
    rows3 = sg.sweep(3.0, dmin=1e-7, dmax=1e-3, n=5)
    rows1 = sg.sweep(-1.0, dmin=1e-7, dmax=1e-3, n=5)
    assert sg.fitted_slope(rows3, 1e-7, 1e-3) == pytest.approx(-0.5, abs=0.02)
    assert sg.fitted_slope(rows1, 1e-7, 1e-3) == pytest.approx(-1.5, abs=0.02)
    # a_q d^(-1/2) -> 1 at q = -1 and a_q d^(1/2) -> 1 at q = 3
    assert float(sg.mp_weight(1, -1 + 1e-10)[0]) * 1e5 == pytest.approx(1.0, abs=1e-4)
    assert float(sg.mp_weight(1, 3 - 1e-10)[0]) * 1e-5 == pytest.approx(1.0, abs=1e-4)


def test_removable_values_have_bounded_gain():
    for q0 in (1.0, 2.0):
        g = [sg.gain(q0 + s * d) for d in (1e-2, 1e-6, 1e-10) for s in (-1, 1)]
        assert max(g) / min(g) < 1.01


def test_closed_form_matrix_fails_where_the_article_says():
    # overflow near q = 1, loss of accuracy near q = 2 and q = 3
    ref1 = sg.mp_curve(1 + 1e-4)
    assert np.isnan(sg.rel_err(sg.seg_closed(1 + 1e-4), ref1))
    ref2 = sg.mp_curve(2 + 1e-8)
    assert sg.rel_err(sg.seg_closed(2 + 1e-8), ref2) > 1e-9
    assert sg.rel_err(sg.seg_product(2 + 1e-8), ref2) < 1e-13


def test_fig2_curve_size_grows_like_the_text_says():
    s1, _ = sg.extent(1.0)
    s3, m3 = sg.extent(2.9999)
    sm, mm = sg.extent(-0.98)
    assert 30 < s3 / s1 < 60 and m3 > 100      # "more than 30 times", coordinates above 125
    assert sm / s1 > 100


def test_asymptotic_constants():
    c3, cm = sg.asymptotic_constants()
    # phi_1 ~ c * d^(-1/2) (q -> 3) and ~ c * d^(-3/2) (q -> -1)
    d = 1e-8
    assert sg.sup_phi(3 - d, 1) * d ** 0.5 == pytest.approx(c3, rel=2e-2)
    assert sg.sup_phi(-1 + d, 1) * d ** 1.5 == pytest.approx(cm, rel=2e-2)


def test_spline_validates_its_input():
    from xhermite import QHermiteSpline1D
    with pytest.raises(ValueError):
        QHermiteSpline1D([0, 1, 1], [0, 1, 2])
    with pytest.raises(ValueError):
        QHermiteSpline1D([0, 1, 2], [0, 1])
    with pytest.raises(ValueError):
        QHermiteSpline1D([0, 1, 2], [0, 1, 2], [0, 1])
