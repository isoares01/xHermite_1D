"""Figs. 1-5 are regenerated from the original notebook data."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from xhermite.figures import FIG2_X, FIG2_Y, build_figures
from xhermite.mathematica_reference import xhermite_segment


def test_build_figures_writes_all_files(tmp_path):
    files = build_figures(str(tmp_path))
    assert len(files) == 15
    assert "fig3_weierstrass.pdf" in files and "fig5b_per_segment.pdf" in files


@pytest.mark.parametrize("q", [-0.98, -0.6, 1.0, 2.6, 2.98, 2.9999])
def test_fig2_curve_interpolates_the_control_data(q):
    """Every segment of Fig. 2 passes through the prescribed points P_i."""
    for i in range(3):
        for P, V in (FIG2_X, FIG2_Y):
            s = xhermite_segment(np.array([0.0, 1.0]), q, P[i], P[i + 1], V[i], V[i + 1])
            assert s[0] == pytest.approx(P[i], abs=1e-9)
            assert s[1] == pytest.approx(P[i + 1], rel=1e-9, abs=1e-9)
