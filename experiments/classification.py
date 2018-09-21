from __future__ import division, print_function

import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from torch.distributions import Bernoulli

from ngboost import *
from ngboost.scores import *
from experiments.evaluation import *
from ngboost.distns import get_categorical_distn, get_beta_distn


np.random.seed(123)


if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    sb = NGBoost(Base = default_tree_learner,
                 Dist = get_categorical_distn(3),
                 Score = Brier,
                 n_estimators = 10,
                 learning_rate = 0.05,
                 natural_gradient = True,
                 second_order = True,
                 quadrant_search = False,
                 normalize_inputs=True,
                 normalize_outputs=False,
                 minibatch_frac = 0.2,
                 nu_penalty=1e-5,
                 tol=1e-4)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    sb.fit(X_train, y_train)

    preds = sb.pred_dist(X_test)
    print(preds.probs)

    #c_stat = calculate_concordance_naive(preds.icdf(torch.tensor(0.5)),
    #                                     T_test, 1 - Y_test)
    #pred, obs, slope, intercept = calibration_time_to_event(preds, T_test, 1 - Y_test)
    #print("C stat: %.4f" % c_stat)
    #print("Censorship rate:", 1-np.mean(sprint["y"]))
    #print("True median [uncens]:", np.median(T_test[Y_test == 1]))
    #print("True median [cens]:", np.median(T_test[Y_test == 0]))
    #print("Pred median:", preds.icdf(torch.tensor(0.5)).mean().detach().numpy())
    #print("Calibration slope: %.4f, intercept: %.4f" % (slope, intercept))

