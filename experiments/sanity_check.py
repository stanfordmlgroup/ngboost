import numpy as np

from torch.distributions.log_normal import LogNormal
from sklearn.tree import DecisionTreeRegressor
from ngboost import SurvNGBoost
from experiments.evaluation import calculate_concordance_naive

from ngboost.scores import MLE_surv, CRPS_surv
from experiments.sim_experiment import *


def main():
    #m, n = 100, 50
    #X = np.random.rand(m, n).astype(np.float32)
    # Y = np.random.rand(m).astype(np.float32) * 2 + 1
    #Y = np.sum(X, axis=1)
    #Y = (Y - np.mean(Y)) / np.std(Y)
    #Y = Y - np.min(Y) + 1e-2
    #C = (np.random.rand(m) > 0.5).astype(np.float32)
    # C = np.zeros_like(Y)
    X = simulate_X(num_unif=30, num_bi=30, N=1000, num_normal=30, normal_cov_strength=[0.5]*30)
    Y, C = simulate_Y_C(X)
    print('Censoring fraction: %f' % ((np.sum(C) / len(C))))

    sb = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='mse'),
                     Dist = LogNormal,
                     Score = MLE_surv,
                     n_estimators = 12,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     nu_penalty=1e-5)

    train, test = sb.fit(X[:700], Y[:700], C[:700], X[700:], Y[700:], C[700:])
    preds = sb.pred_mean(X)

    print("Train/DecTree:", calculate_concordance_naive(preds, Y, C))
    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds), np.mean(Y)))

if __name__ == '__main__':
    main()