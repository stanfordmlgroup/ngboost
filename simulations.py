import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from survboost import SurvBoost
from scoring_rules import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from torch.distributions.log_normal import LogNormal, Normal
from evaluation import *


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
    
    
def eval_preds(filename):
    preds = np.load(filename)
    df = pd.read_csv("data/simulated/sim_data_test.csv")
    print(calculate_concordance_naive(preds, df["Y"], df["C"]))
    # print(calculate_concordance_dead_only(preds, df["Y"], df["C"]))
    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds), np.mean(df["Y"])))
    
    
def run_survboost():

    df_train = pd.read_csv("data/simulated/sim_data_train.csv")
    df_test = pd.read_csv("data/simulated/sim_data_test.csv")

    sb = SurvBoost(Base = lambda: DecisionTreeRegressor(criterion='mse'),
                   Dist = LogNormal,
                   Score = CRPS_surv,
                   n_estimators = 200)

    sb.fit(df_train.drop(["Y", "C"], axis=1).as_matrix(), 
           df_train["Y"], df_train["C"])
    preds_test = sb.pred_mean(df_test.drop(["Y", "C"], axis=1))
    np.save("data/simulated/sim_preds_survboost.npy", preds_test)

    
if __name__ == "__main__":
    
    Y, X, beta = gen_sim_data(N=1000, var_explained = 0.95)
    CENSORED_FRAC = 0.5
    C = np.zeros_like(Y)
    C[:int(CENSORED_FRAC * len(Y))] = 1
    df = pd.DataFrame(X, columns=["X%d" % i for i in range(X.shape[1])])
    df["Y"] = Y
    df["C"] = C
    df = df.sample(frac=1, replace=False)
    df.iloc[:500].to_csv("data/simulated/sim_data_train.csv", index=False)
    df.iloc[500:].to_csv("data/simulated/sim_data_test.csv", index=False)
    
