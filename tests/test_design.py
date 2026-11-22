"""Section 4.4: geometric illustration of the invariances (module xhermite.design).

Encodes the statements of Remark "Local frames" and Fig. "design":
* the centered variant is translation covariant, constant-preserving and reversal symmetric;
* the left-anchored variant is translation covariant but NOT reversal symmetric;
* anchoring at the mean level while keeping the original weight is NOT reversal symmetric
  (both the mean-level anchor and the even weight are needed);
* the basic form has none of the three for q != 1.
"""
import numpy as np
import pytest

from xhermite import design


@pytest.mark.parametrize("q", [-0.5, 0.5, 2.0, 2.6])
def test_translation_covariance(q):
    assert design.translation_deviation("centered", q) < 1e-12
    assert design.translation_deviation("left", q) < 1e-12
    assert design.translation_deviation("raw", q) > 1e-2


@pytest.mark.parametrize("q", [-0.5, 0.5, 2.0, 2.6])
def test_reversal_symmetry_needs_both_ingredients(q):
    assert design.reversal_deviation("centered", q) < 1e-12
    assert design.reversal_deviation("left", q) > 1e-2
    assert design.reversal_deviation("midraw", q) > 1e-2
    assert design.reversal_deviation("raw", q) > 1e-2


@pytest.mark.parametrize("scheme", ["raw", "left", "midraw", "centered"])
def test_classical_limit_and_hermite_data(scheme):
    c1 = design.curve(scheme, 1.0)
    assert np.allclose(c1, design.curve("centered", 1.0), atol=1e-12)
    for q in (-0.5, 0.5, 2.0):
        c = design.curve(scheme, q)
        assert np.allclose(c[0], design.P0, atol=1e-9) and np.allclose(c[-1], design.P1, atol=1e-9)


def test_constants_reproduced_by_variants_only():
    t = design.T
    c = 7.0
    for q in (0.5, 2.0):
        assert np.allclose(design.centered_segment(t, q, c, c), c, atol=1e-12)
        assert np.allclose(design.left_anchored_segment(t, q, c, c), c, atol=1e-12)
        assert not np.allclose(design.xhermite_segment(t, q, c, c), c, atol=1e-3)


def test_quoted_blending_deviation_below_two_percent_for_basic_form():
    """The Conclusions state < 2 % for 0.5 <= q <= 1.5 (basic form); the centered variant is ~2 %."""
    qs = np.round(np.arange(0.5, 1.5001, 0.05), 2)
    assert max(design.blending_deviation(q, "raw") for q in qs) < 0.02
    assert max(design.blending_deviation(q, "centered") for q in qs) < 0.0205


def test_build_writes_figure_and_macros(tmp_path):
    pytest.importorskip("matplotlib")
    path, res = design.build(str(tmp_path))
    assert path.endswith("fig_design.pdf")
    assert (tmp_path / "results" / "macros_design.tex").exists()
    assert res["TransCen"] < 1e-12 and res["RevCen"] < 1e-12
