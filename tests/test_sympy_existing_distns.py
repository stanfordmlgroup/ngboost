"""Prove the SymPy factory reproduces existing hand-written score classes.

For each existing distribution, we pass just a ``sympy.stats`` distribution
to ``make_sympy_log_score`` (no manual score expression), and verify
``score()``, ``d_score()``, and ``metric()`` match the hand-written
implementations numerically.

This demonstrates that ``make_sympy_log_score`` *could have* replaced all
existing hand-written score classes — the user only needs to define the
distribution, and the factory does everything.
"""

import numpy as np
import pytest
import sympy as sp
import sympy.stats as symstats

from ngboost.distns import Gamma, Normal, Poisson
from ngboost.distns.sympy_utils import make_sympy_log_score
from ngboost.manifold import manifold
from ngboost.scores import LogScore

# ---------------------------------------------------------------------------
# SymPy-generated score classes — just define the distribution, nothing else
# ---------------------------------------------------------------------------

# --- Normal (2 params: loc [identity], scale [log]) ---
_loc, _scale, _y_norm = sp.symbols("loc scale y", positive=True)
SympyNormalLogScore = make_sympy_log_score(
    params=[(_loc, False), (_scale, True)],
    y=_y_norm,
    sympy_dist=symstats.Normal("Y", _loc, _scale),
    name="SympyNormalLogScore",
)

# --- Gamma (shape-rate parameterisation: alpha [log], beta [log]) ---
_alpha_g, _beta_g, _y_gamma = sp.symbols("alpha beta y", positive=True)
SympyGammaLogScore = make_sympy_log_score(
    params=[(_alpha_g, True), (_beta_g, True)],
    y=_y_gamma,
    sympy_dist=symstats.Gamma("Y", _alpha_g, 1 / _beta_g),
    name="SympyGammaLogScore",
)

# --- Poisson (1 param: mu [log]) ---
_mu_p, _y_pois = sp.symbols("mu y", positive=True)
SympyPoissonLogScore = make_sympy_log_score(
    params=[(_mu_p, True)],
    y=_y_pois,
    sympy_dist=symstats.Poisson("Y", _mu_p),
    name="SympyPoissonLogScore",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hw_and_sympy(dist_cls, sympy_score_cls, params):
    """Create a hand-written manifold instance and a matching SymPy score obj."""
    M = manifold(LogScore, dist_cls)
    D_hw = M(params)

    D_sy = sympy_score_cls.__new__(sympy_score_cls)
    D_sy.__dict__.update(D_hw.__dict__)
    return D_hw, D_sy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


class TestNormal:
    """Verify SymPy-generated Normal LogScore matches hand-written."""

    @pytest.fixture
    def objs(self):
        params = np.array([[1.0], [np.log(2.0)]])
        return _make_hw_and_sympy(Normal, SympyNormalLogScore, params)

    def test_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.score(Y), sy.score(Y))

    def test_d_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.d_score(Y), sy.d_score(Y))

    def test_metric(self, objs):
        hw, sy = objs
        assert np.allclose(hw.metric(), sy.metric())


class TestGamma:
    """Verify SymPy-generated Gamma LogScore matches hand-written."""

    @pytest.fixture
    def objs(self):
        params = np.array([[np.log(2.0)], [np.log(1.5)]])
        return _make_hw_and_sympy(Gamma, SympyGammaLogScore, params)

    def test_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.score(Y), sy.score(Y))

    def test_d_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.d_score(Y), sy.d_score(Y))

    def test_metric(self, objs):
        hw, sy = objs
        assert np.allclose(hw.metric(), sy.metric())


class TestPoisson:
    """Verify SymPy-generated Poisson LogScore matches hand-written."""

    @pytest.fixture
    def objs(self):
        params = np.array([[np.log(3.0)]])
        return _make_hw_and_sympy(Poisson, SympyPoissonLogScore, params)

    def test_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.score(Y), sy.score(Y))

    def test_d_score(self, objs):
        hw, sy = objs
        Y = hw.sample(5)
        assert np.allclose(hw.d_score(Y), sy.d_score(Y))

    def test_metric(self, objs):
        hw, sy = objs
        assert np.allclose(hw.metric(), sy.metric())
