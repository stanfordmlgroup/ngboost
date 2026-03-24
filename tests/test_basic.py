from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal


# TODO: This is non-deterministic in the model fitting
def test_classification(breast_cancer_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        log_loss,
        roc_auc_score,
    )

    x_train, x_test, y_train, y_test = breast_cancer_data
    ngb = NGBClassifier(Dist=Bernoulli, verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = roc_auc_score(y_test, preds)

    # loose score requirement so it isn't failing all the time
    assert score >= 0.85

    preds = ngb.predict_proba(x_test)
    score = log_loss(y_test, preds)
    assert score <= 0.30

    score = ngb.score(x_test, y_test)
    assert score <= 0.30

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Bernoulli)

    score = roc_auc_score(y_test, preds[:, 1])

    assert score >= 0.85


# TODO: This is non-deterministic in the model fitting
def test_regression(california_housing_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        mean_squared_error,
    )

    x_train, x_test, y_train, y_test = california_housing_data
    ngb = NGBRegressor(verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = mean_squared_error(y_test, preds)
    assert score <= 15

    score = ngb.score(x_test, y_test)
    assert score <= 15

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Normal)

    score = mean_squared_error(y_test, preds)
    assert score <= 15


def test_classifier_validation_fraction_is_supported(breast_cancer_data):
    x_train, x_test, y_train, _ = breast_cancer_data
    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=25,
        verbose=False,
        random_state=1,
        validation_fraction=0.2,
        early_stopping_rounds=2,
    )

    assert ngb.get_params()["validation_fraction"] == 0.2
    assert ngb.get_params()["early_stopping_rounds"] == 2

    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    assert preds.shape[0] == x_test.shape[0]
