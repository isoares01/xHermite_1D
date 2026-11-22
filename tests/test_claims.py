"""Qualitative claims of the article that carry no generated number (Sections 3.1, 4.2, 5.3)."""
import sympy as sp
from xhermite import claims


def test_polynomial_and_rational_weights():
    f = claims.weight_forms()
    assert f[sp.Integer(0)] == ("polynomial", 4)          # w_0 = 1 - t^4/4
    assert f[sp.Rational(1, 2)] == ("polynomial", 8)      # q = 1 - 1/2
    assert f[sp.Rational(2, 3)] == ("polynomial", 12)     # q = 1 - 1/3
    assert f[sp.Integer(2)] == ("rational", 4)            # w_2 = 1/(1 - t^4/4)
    assert f[sp.Rational(3, 2)] == ("rational", 8)
    assert f[sp.Rational(4, 3)] == ("rational", 12)


def test_inverse_weight_t4_coefficient():
    q = sp.symbols("q")
    assert sp.simplify(claims.t4_coefficient_of_inverse_weight() + (q - 1) / 4) == 0


def test_w4_aliasing_constants():
    ce, co, spread, residues = claims.aliasing_constants()
    assert abs(ce - 0.323) < 5e-4 and abs(co - 0.314) < 5e-4
    assert spread < 1e-9
    assert residues == [1, 17]
