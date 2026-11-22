"""
Weierstrass-type benchmark functions (Section 5 of the article, Eq. (weierstrass_def)).

    W_a(x) = sum_{k=1}^{infty} sin(pi k^a x) / (pi k^a),   a = 2, 3, 4, 5

truncated at K = 250 terms.

The article's benchmark uses ZERO tangent restrictions (see
``xhermite.benchmark``): W_a has no derivative in the limit K -> infinity,
and the term-by-term derivative of the truncated series,

    d/dx W_a(x) ~ sum_{k=1}^{K} cos(pi k^a x),

grows linearly with K (it equals K at x = 0), so it is NOT a meaningful
tangent. ``weierstrass_derivative`` is kept only to document this.
For exact evaluation at rational abscissae use
``xhermite.mathematica_reference.weierstrass_rational``.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

DEFAULT_K = 250

# NOTE on high-frequency content: with the default K=250 and a in
# {2,3,4,5}, the highest-order term has "frequency" k^a up to 250**5,
# so evaluating/plotting W_a (and especially its derivative) requires a
# correspondingly dense x-grid to avoid aliasing; a coarse grid will
# visually alias the high-k terms into spurious low-frequency ripple.
# The RMSE benchmark of Section 5 uses this K on evaluation grids of a
# few hundred points, consistent with the article; finer visual
# inspection of W_a itself may need denser sampling.


def weierstrass_function(x: np.ndarray, a: float, K: int = DEFAULT_K) -> np.ndarray:
    """W_a(x) of Eq. (weierstrass_def), truncated at K terms."""
    x = np.asarray(x, dtype=float)
    k = np.arange(1, K + 1, dtype=float).reshape(-1, *([1] * x.ndim))
    ka = k ** a
    return np.sum(np.sin(np.pi * ka * x) / (np.pi * ka), axis=0)


def weierstrass_derivative(x: np.ndarray, a: float, K: int = DEFAULT_K) -> np.ndarray:
    """
    Term-by-term derivative sum_{k=1}^{K} cos(pi k^a x) of the truncated series.

    NOT USED BY THE ARTICLE PROTOCOL (which uses zero tangents, see
    ``xhermite.benchmark``). Its terms are undamped, so its magnitude grows
    with K (exactly K at x = 0; at x = 0.4, a = 2: 112, 447, 1789 for
    K = 250, 1000, 4000): it does not converge and is not a meaningful
    tangent for W_a. Kept only to document this fact.
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(1, K + 1, dtype=float).reshape(-1, *([1] * x.ndim))
    ka = k ** a
    return np.sum(np.cos(np.pi * ka * x), axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error (Section 5.1, Error measures)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def oracle_q_search(
    evaluate_fn: Callable[[float], np.ndarray],
    x_eval: np.ndarray,
    y_true: np.ndarray,
    q_grid: Sequence[float],
) -> Tuple[float, float, np.ndarray]:
    """
    Oracle (target-function-aware) search for q_Emin (Section 5.1, Selection of q): the value
    of q in ``q_grid`` minimizing the RMSE between ``evaluate_fn(q)`` and
    ``y_true``. This reproduces the methodology of Section 5 and is
    explicitly acknowledged there (Section 5.6, Discussion) as an oracle procedure not
    available in practice -- see ``heuristics.py`` for oracle-free
    alternatives.

    Returns
    -------
    q_opt : float
    rmse_min : float
    rmse_spectrum : ndarray
        RMSE at every point of q_grid (NaN where evaluation failed,
        e.g. near a singular value of q).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    spectrum = np.full_like(q_grid, np.nan)
    q_opt, rmse_min = None, np.inf

    for idx, q in enumerate(q_grid):
        try:
            y_q = evaluate_fn(float(q))
            err = rmse(y_true, y_q)
        except Exception:
            continue
        spectrum[idx] = err
        if err < rmse_min:
            rmse_min = err
            q_opt = float(q)

    return q_opt, rmse_min, spectrum
