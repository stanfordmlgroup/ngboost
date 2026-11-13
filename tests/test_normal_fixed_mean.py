import numpy as np
import pytest
from scipy.stats import norm

from ngboost import NGBRegressor
from ngboost.distns import NormalFixedMean
from ngboost.manifold import manifold
from ngboost.scores import CRPScore, LogScore


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
