import numpy as np
import pytest

from ngboost import NGBRegressor


@pytest.mark.parametrize("first_rounds", [0, 2])
def test_partial_fit_preserves_initial_distribution(first_rounds):
    rng = np.random.RandomState(7)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(size=40)
    model = NGBRegressor(n_estimators=first_rounds, verbose=False, random_state=1).fit(
        X, y
    )
    initial = model.init_params.copy()
    predictions = model.pred_dist(X)._params.copy()

    model.set_params(n_estimators=0).partial_fit(X, y + 100)

    np.testing.assert_array_equal(model.init_params, initial)
    np.testing.assert_array_equal(model.pred_dist(X)._params, predictions)

    # A fresh fit must still learn the new data's marginal distribution.
    model.fit(X, y + 100)
    fresh = NGBRegressor(n_estimators=0, verbose=False).fit(X, y + 100)
    np.testing.assert_array_equal(model.init_params, fresh.init_params)


def test_partial_fit_with_new_targets_matches_continuation_from_original_intercept():
    rng = np.random.RandomState(14)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(size=40)
    model = NGBRegressor(n_estimators=2, verbose=False, random_state=1).fit(X, y)
    initial = model.init_params.copy()
    observed_initial_params = []

    def monitor(dist, target, weights):
        observed_initial_params.append(dist._params.copy())
        return dist.total_score(target, sample_weight=weights)

    before = model.pred_dist(X)._params.copy()
    model.set_params(n_estimators=1).partial_fit(X, y + 2, train_loss_monitor=monitor)
    np.testing.assert_array_equal(observed_initial_params[0], before)
    np.testing.assert_array_equal(model.init_params, initial)
