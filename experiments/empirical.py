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
from ngboost.scores import MLE_surv, CRPS_surv

if __name__ == "__main__":

    data = load_boston()
    X, y = data["data"], data["target"]
    C = np.zeros(len(y))

#    y = ( y - np.mean(y) ) / np.std(y)
    X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C)

    ngb = SurvNGBoost(Base=lambda: DecisionTreeRegressor(criterion='mse'),
                  Dist=HomoskedasticNormal,
                  Score=MLE_surv,
                  n_estimators=200,
                  learning_rate=0.02,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=True)

    ngb.fit(X_train, y_train, C_train, X_test, y_test, C_test)
    y_pred = ngb.pred_mean(X_test)
    print(r2_score(y_test, y_pred))

    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    print(r2_score(y_test, gbr.predict(X_test)))
