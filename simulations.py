import numpy as np
import scipy as sp
import scipy.stats
from survboost import SurvBoost
from scoring_rules import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from torch.distributions.log_normal import LogNormal, Normal
from evaluation import calculate_concordance_naive


def gen_sim_data(N=50, M=5, var_explained=0.8, noise_dist=sp.stats.norm):
    """
    Generate simulated data, according to the process described in [1].
    
    Noise distribution should be one of:
    [ sp.stats.norm, sp.stats.genextreme, sp.stats.logistic ]
    
    Returns: (Y, X, beta)
    
    [1] Schmid, Matthias, and Torsten Hothorn. 
    Flexible Boosting of Accelerated Failure Time Models.
    BMC Bioinformatics 9 (June 6, 2008): 269. 
    https://doi.org/10.1186/1471-2105-9-269.
    """
    cov_matrix = np.ones((M, M)) * 0.5 + np.eye(M) * 0.5
    covariates = sp.stats.multivariate_normal.rvs(cov=cov_matrix, size=N)
    beta = np.r_[np.array((0.5, 0.25, -0.25, -0.5, 0.5)), np.zeros(M - 5)]
    unnoised = np.dot(covariates, beta)
    sigma = np.sqrt(np.var(unnoised) * (1 / var_explained - 1))
    noisy = unnoised + sigma * noise_dist.rvs(size=N)
    times = np.exp(noisy)
    return (times, covariates, beta)
    
    
if __name__ == "__main__":
    
    sb = SurvBoost(Base = lambda : DecisionTreeRegressor(criterion='mse'),
                   Dist = LogNormal,
                   Score = MLE_surv,
                   n_estimators = 200)
                   
    Y, X, beta = gen_sim_data(N=1000)
    CENSORED_FRAC = 0.9
    C = np.zeros_like(Y)
    C[:int(CENSORED_FRAC * len(Y))] = 1
    sb.fit(X, Y, C)
    preds_train = sb.pred_mean(X)
    
    print("Train/DecTree:", calculate_concordance_naive(preds_train, Y, C))
    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds_train), np.mean(Y)))
    
    Y, X, beta = gen_sim_data(N=1000)
    C = np.zeros_like(Y)    
    preds_test = sb.pred_mean(X)

    print("Test/DecTree:", calculate_concordance_naive(preds_test, Y, C))
    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds_test), np.mean(Y)))
    
    