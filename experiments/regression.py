from __future__ import print_function
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import r2_score
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE_surv, CRPS_surv

if __name__ == "__main__":

    data = load_boston()
    X, y = data["data"], data["target"]
    C = np.zeros(len(y))

    X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C)
    base_learner = lambda: DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)

    print("Heteroskedastic")
    ngb = SurvNGBoost(Base=base_learner,
                  Dist=Normal,
                  Score=CRPS_surv,
                  n_estimators=100,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=False)

    ngb.fit(X_train, y_train, C_train, X_test, y_test, C_test)
    y_pred = ngb.pred_mean(X_test)
    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(mean_squared_error(y_train, ngb.pred_mean(X_train)))
    # print(np.mean(y_pred))

    print("Homoskedastic")
    ngb = SurvNGBoost(Base=base_learner,
                  Dist=HomoskedasticNormal,
                  Score=CRPS_surv,
                  n_estimators=100,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=False)

    ngb.fit(X_train, y_train, C_train, X_test, y_test, C_test)
    y_pred = ngb.pred_mean(X_test)
    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(mean_squared_error(y_train, ngb.pred_mean(X_train)))

    print("Scikit-Learn GBM")
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    print(r2_score(y_test, gbr.predict(X_test)))
    print(mean_squared_error(y_test, gbr.predict(X_test)))
    print(mean_squared_error(y_train, gbr.predict(X_train)))
