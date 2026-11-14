import warnings
from typing import Tuple

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival, distns
from ngboost.distns import (
    Bernoulli,
    Cauchy,
    ClassificationDistn,
    Distn,
    Exponential,
    Gamma,
    HalfNormal,
    LogNormal,
    MultivariateNormal,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    RegressionDistn,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
    k_categorical,
)
from ngboost.manifold import manifold
from ngboost.scores import CRPScore, LogScore, Score

# test all the dist methods and score implementation methods,#
# i.e. they all return proper shapes and sizes and types
# check metric lines up with defaults for lognormal where applicable


Tuple4Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Tuple5Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module")
def regression_data():
    data = fetch_california_housing()
    X, y = data["data"][:1000], data["target"][:1000]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def classification_data():
    data = load_breast_cancer()
    X, y = data["data"][:1000], data["target"][:1000]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist",
    [
        Normal,
        NormalFixedVar,
        NormalFixedMean,
        HalfNormal,
        LogNormal,
        Exponential,
        Gamma,
        # This case fails for both learners, so the marker is strict: if T starts
        # to pass, the run fails and the marker has to go. Only a ValueError
        # counts as the expected failure.
        pytest.param(
            T,
            marks=pytest.mark.xfail(
                reason=(
                    "NGBRegressor(Dist=T) raises a ValueError: the natural "
                    "gradient for log df runs away as df grows, and the "
                    "gradient goes to NaN (#275)"
                ),
                raises=ValueError,
                strict=True,
            ),
        ),
        TFixedDf,
        TFixedDfFixedVar,
        Cauchy,
        Weibull,
    ],
)
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
def test_dists_runs_on_examples_logscore(dist: Distn, learner, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("dist", [Normal, LogNormal, Exponential])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
def test_dists_runs_on_examples_crpscore(dist: Distn, learner, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=CRPScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.parametrize("dist", [LogNormal, Exponential])
@pytest.mark.parametrize("score", [LogScore, CRPScore])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
def test_survival_runs_on_examples(
    dist: Distn, score: Score, learner, california_housing_survival_data
):
    (
        X_train,
        X_test,
        T_surv_train,
        E_surv_train,
        Y_surv_test,
    ) = california_housing_survival_data
    # test early stopping features
    ngb = NGBSurvival(Dist=dist, Score=score, Base=learner, verbose=False)
    ngb.fit(X_train, T_surv_train, E_surv_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
def test_bernoulli(learner, classification_data: Tuple4Array):
    X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = classification_data
    # test early stopping features
    # test other args, n_trees, LR, minibatching- args as fixture
    ngb = NGBClassifier(Dist=Bernoulli, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_cls_train, Y_cls_train)
    y_pred = ngb.predict(X_cls_test)
    y_prob = ngb.predict_proba(X_cls_test)
    y_dist = ngb.pred_dist(X_cls_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("k", [2, 4, 7])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
def test_categorical(k: int, learner, classification_data: Tuple4Array):
    X_train, X_test, y_train, _ = classification_data
    dist = k_categorical(k)
    y_train = np.random.randint(0, k, (len(y_train)))
    # test early stopping features
    ngb = NGBClassifier(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_prob = ngb.predict_proba(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
# Ignore the k=1 warning
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multivariatenormal(k: int, learner):
    dist = MultivariateNormal(k)

    # Generate some sample data
    N = 500
    X_train = np.random.randn(N, k)
    y_fns = [np.sin, np.cos, np.exp]
    y_cols = [
        fn(X_train[:, num_col]).reshape(-1, 1) + np.random.randn(N, 1)
        for num_col, fn in enumerate(y_fns[:k])
    ]
    y_train = np.hstack(y_cols)
    X_test = np.random.randn(N, k)

    ngb = NGBRegressor(
        Dist=dist, Score=LogScore, Base=learner, verbose=False, n_estimators=50
    )
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)

    mean = y_dist.mean
    sample = y_dist.rv()
    scipy_list = y_dist.scipy_distribution()


# Slicing and len. Distn.__getitem__ and Distn.__len__ read self._params, so
# every distribution has to store the params it was built from.

_N_ROWS = 6


def _concrete_distns():
    """The concrete distributions in ngboost.distns.__all__.

    k_categorical and MultivariateNormal are factories, so they are called
    with a small size. Distn, RegressionDistn and ClassificationDistn are
    abstract bases and are left out.
    """
    bases = (Distn, RegressionDistn, ClassificationDistn)
    factories = {
        "k_categorical": lambda: k_categorical(3),
        "MultivariateNormal": lambda: MultivariateNormal(2),
    }
    cases = []
    for name in distns.__all__:
        obj = getattr(distns, name)
        if name in factories:
            obj = factories[name]()
        if obj in bases:
            continue
        cases.append(pytest.param(obj, id=name))
    return cases


def _build(Dist):
    rng = np.random.default_rng(0)
    return Dist(rng.normal(scale=0.3, size=(Dist.n_params, _N_ROWS)))


def _rows(value, key):
    # a few distributions keep a scalar in params, such as the n of BetaBinomial
    return value if np.ndim(value) == 0 else value[key]


def test_concrete_distns_cover_all_exports():
    names = [case.id for case in _concrete_distns()]
    assert len(names) == len(set(names))
    assert {"Normal", "NormalFixedVar", "NormalFixedMean", "Bernoulli"} <= set(names)


@pytest.mark.parametrize("Dist", _concrete_distns())
def test_len(Dist):
    assert len(_build(Dist)) == _N_ROWS


@pytest.mark.parametrize("Dist", _concrete_distns())
def test_slice_keeps_class_and_params(Dist):
    dist = _build(Dist)
    sliced = dist[1:4]
    assert sliced.__class__ is Dist
    assert len(sliced) == 3
    assert sliced.params.keys() == dist.params.keys()
    for name, value in dist.params.items():
        np.testing.assert_allclose(sliced.params[name], _rows(value, slice(1, 4)))


@pytest.mark.parametrize("Dist", _concrete_distns())
def test_integer_index_keeps_class_and_params(Dist):
    dist = _build(Dist)
    single = dist[2]
    assert single.__class__ is Dist
    for name, value in dist.params.items():
        np.testing.assert_allclose(single.params[name], _rows(value, 2))


@pytest.mark.parametrize(
    "Dist, expected_loc, expected_scale",
    [
        (NormalFixedVar, lambda p: p[0], lambda p: np.ones(len(p[0]))),
        (NormalFixedMean, lambda p: np.zeros(len(p[0])), lambda p: np.exp(p[0])),
    ],
    ids=["NormalFixedVar", "NormalFixedMean"],
)
def test_normal_fixed_slice_matches_rows(Dist, expected_loc, expected_scale):
    params = np.random.default_rng(1).normal(size=(1, 8))
    dist = Dist(params)
    rows = slice(2, 7)
    sliced = dist[rows]
    assert sliced.__class__ is Dist
    assert len(dist) == 8
    assert len(sliced) == 5
    np.testing.assert_allclose(sliced.loc, expected_loc(params)[rows])
    np.testing.assert_allclose(sliced.scale, expected_scale(params)[rows])
    np.testing.assert_allclose(sliced.params["loc"], sliced.loc)
    np.testing.assert_allclose(sliced.params["scale"], sliced.scale)
    y = np.linspace(-1, 1, 8)
    np.testing.assert_allclose(
        sliced.dist.logpdf(y[rows]), dist.dist.logpdf(y)[rows], rtol=1e-12
    )


@pytest.mark.parametrize("Dist", [NormalFixedVar, NormalFixedMean])
def test_normal_fixed_fitted_prediction_supports_len_and_slice(Dist):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(size=60)
    ngb = NGBRegressor(Dist=Dist, n_estimators=5, verbose=False).fit(X, Y)
    pred = ngb.pred_dist(X)
    assert len(pred) == 60
    sliced = pred[0:2]
    assert sliced.__class__ is Dist
    assert len(sliced) == 2
    np.testing.assert_allclose(sliced.loc, pred.loc[0:2])
    np.testing.assert_allclose(sliced.scale, pred.scale[0:2])
    np.testing.assert_allclose(pred[0].loc, pred.loc[0])


# Distn.__getitem__ passes one-dimensional params to the constructor for an
# integer key, so dist[i] has no row axis, as with numpy. Categorical and
# MultivariateNormal used to need two-dimensional params.


@pytest.mark.parametrize(
    "Dist", [Bernoulli, k_categorical(3)], ids=["Bernoulli", "k_categorical"]
)
def test_categorical_integer_index_gives_one_row(Dist):
    dist = _build(Dist)
    for i in [2, -1]:
        single = dist[i]
        np.testing.assert_allclose(single.class_probs(), dist.class_probs()[i])
        assert single.predict() == dist.predict()[i]
        samples = single.sample(50)
        assert samples.shape == (50,)
        assert set(samples) <= set(range(Dist.K_))
    np.testing.assert_allclose([d.class_probs() for d in dist], dist.class_probs())


@pytest.mark.parametrize("k", [2, 3])
def test_multivariatenormal_integer_index_gives_one_row(k):
    dist = _build(MultivariateNormal(k))
    Y = np.random.default_rng(3).normal(size=(_N_ROWS, k))
    for i in [2, -1]:
        single = dist[i]
        np.testing.assert_allclose(single.predict(), dist.predict()[i])
        np.testing.assert_allclose(single.cov, dist.cov[i])
        np.testing.assert_allclose(single.logpdf(Y[i]), dist.logpdf(Y)[i])
        np.testing.assert_allclose(
            single.scipy_distribution().logpdf(Y[i]), dist.logpdf(Y)[i]
        )
        assert np.shape(single.sample(20)) == (20, k)
    # one distribution evaluated at several points
    np.testing.assert_allclose(single.logpdf(Y), [single.logpdf(y) for y in Y])


def test_integer_index_of_fitted_classifier_and_mvn():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    clf = NGBClassifier(Dist=k_categorical(3), n_estimators=5, verbose=False)
    clf.fit(X, y)
    pred = clf.pred_dist(X)
    np.testing.assert_allclose(pred[0].class_probs(), clf.predict_proba(X)[0])
    assert pred[0].predict() == clf.predict(X)[0]

    Y = np.column_stack([X[:, 0], X[:, 1]]) + rng.normal(size=(60, 2))
    reg = NGBRegressor(Dist=MultivariateNormal(2), n_estimators=5, verbose=False)
    reg.fit(X, Y)
    pred = reg.pred_dist(X)
    np.testing.assert_allclose(pred[0].predict(), reg.predict(X)[0])
    np.testing.assert_allclose(pred[0].logpdf(Y[0]), pred.logpdf(Y)[0])


def _mvn_with_cov(cov, N=4):
    """N MultivariateNormal rows with means 0, ..., p - 1 and covariance cov"""
    p = cov.shape[0]
    # the internal parameters are the means and the lower triangle of L, where
    # inv(cov) = L @ L.T and the diagonal of L is on the log scale
    L = np.linalg.cholesky(np.linalg.inv(cov))
    L[np.diag_indices(p)] = np.log(L[np.diag_indices(p)])
    theta = np.concatenate([np.arange(p, dtype=float), L[np.tril_indices(p)]])
    return MultivariateNormal(p)(np.tile(theta[:, None], (1, N)))


def test_multivariatenormal_params_cov():
    cov = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 0.5]])
    dist = _mvn_with_cov(cov)
    params = dist.params

    assert list(params) == ["loc", "cov"]
    np.testing.assert_allclose(params["loc"], np.tile([0.0, 1.0, 2.0], (4, 1)))
    # get_chol_factor adds 1e-6 to the diagonal of L, so the match is not exact
    np.testing.assert_allclose(
        params["cov"], np.broadcast_to(cov, (4, 3, 3)), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_array_equal(params["cov"], dist.cov)

    # the names fit scipy.stats.multivariate_normal row by row
    Y = np.random.default_rng(0).normal(size=(4, 3))
    scipy_logpdf = [
        multivariate_normal(mean=params["loc"][i], cov=params["cov"][i]).logpdf(Y[i])
        for i in range(4)
    ]
    np.testing.assert_allclose(scipy_logpdf, dist.logpdf(Y))


def test_multivariatenormal_params_scale_is_deprecated():
    dist = _mvn_with_cov(np.array([[1.0, 0.3], [0.3, 2.0]]))
    params = dist.params

    with pytest.warns(FutureWarning, match=r"Use params\['cov'\]") as record:
        scale = params["scale"]
    assert record[0].filename == __file__
    np.testing.assert_array_equal(scale, params["cov"])

    with pytest.warns(FutureWarning, match=r"Use params\['cov'\]") as record:
        scale = params.get("scale")
    assert record[0].filename == __file__
    np.testing.assert_array_equal(scale, params["cov"])

    # the other keys work as in a plain dict and give no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dict(params.items()).keys() == {"loc", "cov"}
        np.testing.assert_array_equal(params["loc"], dist.loc)
        np.testing.assert_array_equal(params.get("cov"), dist.cov)
        assert params.get("df") is None
        with pytest.raises(KeyError):
            params["df"]  # pylint: disable=pointless-statement


@pytest.mark.parametrize("factor", [0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    "values",
    [
        [-2.0, -1.0, 1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 3.0, 3.0, 3.0],
    ],
    ids=["centered", "shifted", "constant_nonzero"],
)
def test_normal_fixed_mean_marginal_fit(values, factor):
    y = factor * np.array(values)
    params = np.asarray(NormalFixedMean.fit(y)).reshape(1, 1)
    fitted = manifold(LogScore, NormalFixedMean)(params)
    # With the mean fixed at zero, the MLE variance is the second moment.
    expected_scale = np.sqrt(np.mean(y**2))
    np.testing.assert_allclose(fitted.scale, expected_scale)
    np.testing.assert_allclose(params[0], np.log(expected_scale))
    np.testing.assert_allclose(
        fitted.logpdf(y), norm.logpdf(y, loc=0, scale=expected_scale)
    )
    np.testing.assert_allclose(fitted.d_score(y).mean(axis=0), 0, atol=1e-14)
    for scale in [expected_scale / 2, expected_scale * 2]:
        assert fitted.total_score(y) < -norm.logpdf(y, loc=0, scale=scale).mean()


@pytest.mark.parametrize("score", [LogScore, CRPScore])
def test_normal_fixed_mean_regressor_refit(score):
    X = np.arange(4).reshape(-1, 1)
    model = NGBRegressor(
        Dist=NormalFixedMean, Score=score, n_estimators=0, verbose=False
    )
    for y in [np.array([-2.0, -1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])]:
        model.fit(X, y)
        prediction = model.pred_dist(X)
        expected_scale = np.sqrt(np.mean(y**2))
        np.testing.assert_allclose(prediction.loc, 0)
        np.testing.assert_allclose(prediction.scale, expected_scale)
        np.testing.assert_allclose(
            prediction.logpdf(y), norm.logpdf(y, loc=0, scale=expected_scale)
        )


@pytest.mark.parametrize("score", [LogScore, CRPScore])
def test_normal_fixed_mean_first_boosting_round(score):
    X = np.arange(4).reshape(-1, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    starting_scales = []

    def monitor(distribution, targets, weights):
        starting_scales.append(distribution.scale.copy())
        return distribution.total_score(targets, sample_weight=weights)

    model = NGBRegressor(
        Dist=NormalFixedMean,
        Score=score,
        n_estimators=1,
        verbose=False,
        random_state=0,
    ).fit(X, y, train_loss_monitor=monitor)
    assert len(starting_scales) == 1
    np.testing.assert_allclose(starting_scales[0], np.sqrt(np.mean(y**2)))
    assert len(model.base_models) == 1
    assert np.all(np.isfinite(model.pred_dist(X).scale))
