from __future__ import print_function
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal

from ngboost import NGBoost, MLE, CRPS
from experiments.evaluation import r2_score

if __name__ == "__main__":

    data = load_boston()
    X, y = data["data"], data["target"]

#    y = ( y - np.mean(y) ) / np.std(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ngb = NGBoost(Base=lambda: DecisionTreeRegressor(criterion='mse'),
                  Dist=Normal,
                  Score=MLE,
                  n_estimators=100,
                  learning_rate=0.5,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=True)

    ngb.fit(X_train, y_train)
    y_pred = ngb.pred_mean(X_test)
    print(r2_score(y_test, y_pred))

    # gbr = GradientBoostingRegressor()
    # gbr.fit(X_train, y_train)
    # print(r2_score(y_test, gbr.predict(X_test)))
