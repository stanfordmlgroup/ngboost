import numpy as np

from torch.distributions.log_normal import LogNormal
from torch.distributions import Exponential
from sklearn.tree import DecisionTreeRegressor
from ngboost import SurvNGBoost
from experiments.evaluation import calculate_concordance_naive

from ngboost.scores import MLE_surv, CRPS_surv
from distns.homoskedastic_normal import HomoskedasticNormal
from experiments.sim_experiment import *

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def main():
    m, n = 1000, 5
    X = np.random.rand(m, n).astype(np.float32) + 1
    #Y = np.random.rand(m).astype(np.float32) * 2 + 1
    Y = np.sum(X, axis=1)
    #Y = (Y - np.mean(Y)) / np.std(Y)
    #Y = Y - np.min(Y) + 1e-2
    C = (np.random.rand(m) > 1.5).astype(np.float32)
    # C = np.zeros_like(Y)
    #X = simulate_X(num_unif=30, num_bi=30, N=1000, num_normal=30, normal_cov_strength=[0.5]*30)
    #Y, C = simulate_Y_C(X)
    print(X.shape, Y.shape, C.shape)
    print('Censoring fraction: %f' % (np.mean(C)))

    sb1 = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3),
                     Dist = HomoskedasticNormal,
                     Score = MLE_surv,
                     n_estimators = 100,
                     learning_rate = 0.1,
                     natural_gradient = False,
                     second_order = False,
                     quadrant_search = False,
                     nu_penalty=1e-5)

    sb2 = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3),
                     Dist = HomoskedasticNormal,
                     Score = MLE_surv,
                     n_estimators = 100,
                     learning_rate = 0.1,
                     natural_gradient = False,
                     second_order = True,
                     quadrant_search = False,
                     nu_penalty=1e-5)

    sb3 = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3),
                     Dist = HomoskedasticNormal,
                     Score = MLE_surv,
                     n_estimators = 100,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = False,
                     quadrant_search = False,
                     nu_penalty=1e-5)

    sb4 = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3),
                     Dist = HomoskedasticNormal,
                     Score = MLE_surv,
                     n_estimators = 100,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     nu_penalty=1e-5)

    gbm = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1)

    train, test = sb1.fit(X[:700], Y[:700], C[:700], X[700:], Y[700:], C[700:])
    train, test = sb2.fit(X[:700], Y[:700], C[:700], X[700:], Y[700:], C[700:])
    train, test = sb3.fit(X[:700], Y[:700], C[:700], X[700:], Y[700:], C[700:])
    train, test = sb4.fit(X[:700], Y[:700], C[:700], X[700:], Y[700:], C[700:])

    preds1 = sb1.pred_mean(X)
    preds2 = sb2.pred_mean(X)
    preds3 = sb3.pred_mean(X)
    preds4 = sb4.pred_mean(X)

    gbm = gbm.fit(X[:700], Y[:700])
    gbm_preds = gbm.predict(X)

    print("Train/NGB-1:", calculate_concordance_naive(preds1[:700], Y[:700], C[:700]), mean_squared_error(preds1[:700], Y[:700]))
    print("Test/NGB-1:", calculate_concordance_naive(preds1[700:], Y[700:], C[700:]), mean_squared_error(preds1[:700], Y[:700]))
    print('---')
    print("Train/NGB-2:", calculate_concordance_naive(preds2[:700], Y[:700], C[:700]), mean_squared_error(preds2[:700], Y[:700]))
    print("Test/NGB-2:", calculate_concordance_naive(preds2[700:], Y[700:], C[700:]), mean_squared_error(preds2[:700], Y[:700]))
    print('---')
    print("Train/NGB-1:", calculate_concordance_naive(preds3[:700], Y[:700], C[:700]), mean_squared_error(preds3[:700], Y[:700]))
    print("Test/NGB-1:", calculate_concordance_naive(preds3[700:], Y[700:], C[700:]), mean_squared_error(preds2[:700], Y[:700]))
    print('---')
    print("Train/NGB-1:", calculate_concordance_naive(preds4[:700], Y[:700], C[:700]), mean_squared_error(preds4[:700], Y[:700]))
    print("Test/NGB-1:", calculate_concordance_naive(preds4[700:], Y[700:], C[700:]), mean_squared_error(preds2[:700], Y[:700]))
    print('---')
    print("Train/GBM:", calculate_concordance_naive(gbm_preds[:700], Y[:700], C[:700]), mean_squared_error(gbm_preds[:700], Y[:700]))
    print("Test/GBM:", calculate_concordance_naive(gbm_preds[700:], Y[700:], C[700:]), mean_squared_error(gbm_preds[:700], Y[:700]))

    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds), np.mean(Y)))

if __name__ == '__main__':
    main()
