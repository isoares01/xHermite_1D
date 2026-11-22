"""
xhermite
========

Open-source companion implementation for:

    Soares, I. J. A.; Costa, A. A.; Murta Junior, L. O.
    "x-Hermite interpolation: a Tsallis-deformed generalization of
    cubic Hermite interpolation."
    (submitted to the Journal of Computational and Applied Mathematics)

This package reproduces, symbolically and numerically, every derivation,
proposition, and numerical experiment reported in the article: the
third-degree x-Hermite basis (Eq. (basis)), the q-dependent transformation
matrix M_q (Appendix A), the linear-independence result (Theorem 3(i)),
the classical limit q -> 1, the Weierstrass-function benchmark
(Section 5), and the boundary/complex-region illustration of
Appendix B ("Behavior beyond the admissible interval").

See ``README.md`` for how each article section maps to a module/example.
"""

from .basis import (
    h0_q_numeric,
    q_hermite_basis_numeric,
    compute_Mq_matrix,
    M1,
    MH_canonical,
    ADMISSIBLE_Q_MIN,
    ADMISSIBLE_Q_MAX,
    ADMISSIBLE_Q_EXCLUDED,
    is_admissible,
)
from .spline import QHermiteSpline1D
from .weierstrass import weierstrass_function, weierstrass_derivative, rmse, oracle_q_search
from .mathematica_reference import xhermite_segment, weierstrass_rational
from .benchmark import PROTOCOL, additivity, constant_reproduction, run_case

# x-Hermite names (the article's nomenclature). The original q-prefixed names are
# kept for backward compatibility; both refer to the same objects.
XHermiteSpline1D = QHermiteSpline1D
qhermite_segment = xhermite_segment   # backward-compatible name
x_hermite_basis_numeric = q_hermite_basis_numeric

__all__ = [
    "XHermiteSpline1D",
    "xhermite_segment",
    "x_hermite_basis_numeric",
    "h0_q_numeric",
    "q_hermite_basis_numeric",
    "compute_Mq_matrix",
    "M1",
    "MH_canonical",
    "ADMISSIBLE_Q_MIN",
    "ADMISSIBLE_Q_MAX",
    "ADMISSIBLE_Q_EXCLUDED",
    "is_admissible",
    "QHermiteSpline1D",
    "weierstrass_function",
    "weierstrass_derivative",
    "rmse",
    "oracle_q_search",
    "qhermite_segment",
    "weierstrass_rational",
    "PROTOCOL",
    "constant_reproduction",
    "additivity",  # deprecated alias of constant_reproduction
    "run_case",
]

__version__ = "1.2.0"
