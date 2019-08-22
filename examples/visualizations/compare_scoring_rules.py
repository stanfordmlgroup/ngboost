import numpy as np
import scipy as sp
import scipy.stats
from ngboost.distns import Normal, Laplace
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS, MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    m, n = 1000, 10
    noise = sp.stats.laplace.rvs(size=(m, 1))
    beta1 = np.random.randn(n, 1)
    beta2 = np.random.randn(n, 1)
    X = np.random.randn(m, n) / np.sqrt(n)
    # Y = X @ beta + 0.5 * noise
    Y = X @ beta1 + 0.5 * np.sqrt(np.exp(X @ beta2)) * noise
    print(X.shape, Y.shape)

    axis = np.linspace(0.0, 2, 200)
    plt.figure(figsize = (8, 3))

    ngb = NGBoost(n_estimators=100, learning_rate=1.0,
                  Dist=Normal,
                  Base=default_linear_learner,
                  natural_gradient=True,
                  minibatch_frac=1.0,
                  Score=CRPS())
    ngb.fit(X, Y)
    preds = ngb.pred_dist(X)
    print(preds.scale.mean())
    print(preds.scale.std())
    pctles, observed, slope, intercept = calibration_regression(preds, Y)

    plt.subplot(1, 2, 1)
    plot_pit_histogram(pctles, observed, label = "CRPS", linestyle = "--")
    plt.subplot(1, 2, 2)
    plt.plot(axis, gaussian_kde(preds.scale)(axis),
             linestyle = "--", color = "black", label = "CRPS")

    ngb = NGBoost(n_estimators=100, learning_rate=0.5,
                  Dist=Normal,
                  Base=default_linear_learner,
                  natural_gradient=True,
                  minibatch_frac=1.0,
                  Score=MLE())
    ngb.fit(X, Y)
    preds = ngb.pred_dist(X)
    print(preds.scale.mean())
    print(preds.scale.std())
    pctles, observed, slope, intercept = calibration_regression(preds, Y)
    plt.subplot(1, 2, 1)
    plot_pit_histogram(pctles, observed, label="MLE", linestyle = "-")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(axis, gaussian_kde(preds.scale)(axis),
             linestyle = "-", color = "black", label = "MLE")
    plt.title("Distribution of predicted $\\sigma$")
    plt.xlabel("$\\sigma$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/compare_scoring_rules.pdf")
    plt.show()
