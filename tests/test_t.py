import numpy as np
import pytest
from scipy import stats

from ngboost.distns import Cauchy, T, TFixedDf, TFixedDfFixedVar
from ngboost.distns.cauchy import CauchyFixedVar


def test_t_params_include_df():
    params = np.array(
        [[0.5, -1.0], [np.log(2.0), np.log(0.5)], [np.log(4.0), np.log(10.0)]]
    )
    dist = T(params)
    assert set(dist.params) == {"loc", "scale", "df"}
    np.testing.assert_allclose(dist.params["df"], [4.0, 10.0])
    Y = np.array([0.0, 1.0])
    np.testing.assert_allclose(stats.t(**dist.params).logpdf(Y), dist.logpdf(Y))


class TFixedDf5(TFixedDf):
    fixed_df = 5.0


class TFixedDf5FixedVar(TFixedDfFixedVar):
    fixed_df = 5.0


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module")
def skewed_sample():
    # skewed and with scale far from 1, so the starting fit depends on both df and
    # on whether the scale is fixed
    return np.random.default_rng(0).exponential(scale=3.0, size=500)


@pytest.mark.parametrize("dist", [TFixedDf, Cauchy, TFixedDf5])
def test_fixed_df_fit_uses_class_df(dist, skewed_sample):
    _, loc, scale = stats.t.fit(skewed_sample, fdf=dist.fixed_df)
    np.testing.assert_allclose(dist.fit(skewed_sample), [loc, np.log(scale)])


def test_cauchy_fit_is_cauchy_mle():
    Y = stats.cauchy(loc=2.0, scale=0.5).rvs(2000, random_state=0)
    loc, scale = stats.cauchy.fit(Y)
    np.testing.assert_allclose(Cauchy.fit(Y), [loc, np.log(scale)], atol=1e-3)


@pytest.mark.parametrize("dist", [TFixedDfFixedVar, CauchyFixedVar, TFixedDf5FixedVar])
def test_fixed_df_fixed_var_fit_uses_class_df_and_unit_scale(dist, skewed_sample):
    _, loc, _ = stats.t.fit(skewed_sample, fdf=dist.fixed_df, fscale=1.0)
    np.testing.assert_allclose(dist.fit(skewed_sample), loc)
