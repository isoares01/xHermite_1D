"""
Regression tests: the Python translation must reproduce the numbers printed
by the original Mathematica notebook that generated the 2013 tables.
"""
import numpy as np
import pytest

from xhermite.basis import compute_Mq_matrix, q_hermite_basis_numeric
from xhermite.mathematica_reference import (
    MATHEMATICA_OUTPUTS,
    midpoint_weights,
    q_grid_mathematica,
    xhermite_segment,
    run_configuration,
)


def test_q_grid_matches_notebook_values():
    g = q_grid_mathematica()
    # q values printed by the notebook are exact members of the grid
    for (q, *_rest) in MATHEMATICA_OUTPUTS.values():
        assert np.min(np.abs(g - q)) < 1e-12
    first_round1 = next(q for q in g if round(q) == 1)
    assert abs(first_round1 - 0.5000013) < 1e-9


@pytest.mark.parametrize("q", [-0.7, 0.4, 1.6, 2.4, 2.8])
def test_stable_segment_equals_appendix_matrix_form(q):
    R = np.array([0.3, 1.7, -0.5, 2.1])
    t = np.linspace(0, 1, 11)
    c = R @ compute_Mq_matrix(q)
    h = q_hermite_basis_numeric(t, q)
    s_matrix = sum(ci * hi for ci, hi in zip(c, h))
    s_stable = xhermite_segment(t, q, *R)
    assert np.allclose(s_matrix, s_stable, rtol=1e-9, atol=1e-10)


def test_removable_at_q1_and_q2():
    for q0 in (1.0, 2.0):
        a0, b0 = midpoint_weights(q0)
        for d in (1e-4, -1e-4):
            a, b = midpoint_weights(q0 + d)
            assert abs(a - a0) < 1e-3 and abs(b - b0) < 1e-3
    assert midpoint_weights(1.0) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("key", sorted(MATHEMATICA_OUTPUTS))
def test_reproduces_notebook_row(key):
    a, ni = key
    q_nb, Eq_nb, Ec_nb, ratio_nb = MATHEMATICA_OUTPUTS[key]
    r = run_configuration(a, ni)
    assert r["q_Emin"] == pytest.approx(q_nb, abs=1e-9)
    assert r["E_q"] == pytest.approx(Eq_nb, rel=1e-7)
    assert r["E_c_notebook"] == pytest.approx(Ec_nb, rel=1e-7)
    assert r["ratio_notebook"] == pytest.approx(ratio_nb, rel=1e-7)
