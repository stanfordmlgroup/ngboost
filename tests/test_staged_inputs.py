import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ngboost import NGBClassifier, NGBRegressor


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("input_type", ["list", "dataframe", "csr"])
def test_staged_predictions_accept_prediction_input_types(classifier, input_type):
    rng = np.random.RandomState(12)
    X = rng.normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(int) if classifier else X[:, 0] + rng.normal(size=30)
    estimator = NGBClassifier if classifier else NGBRegressor
    model = estimator(n_estimators=3, verbose=False, random_state=1).fit(X, y)
    converted = {
        "list": X.tolist(),
        "dataframe": pd.DataFrame(X),
        "csr": sparse.csr_matrix(X),
    }[input_type]

    stages = model.staged_predict(converted)
    assert len(stages) == len(model.base_models)
    for iteration, prediction in enumerate(stages, start=1):
        np.testing.assert_allclose(prediction, model.predict(X, max_iter=iteration))
    if classifier:
        for iteration, probabilities in enumerate(
            model.staged_predict_proba(converted), start=1
        ):
            np.testing.assert_allclose(
                probabilities, model.predict_proba(X, max_iter=iteration)
            )
