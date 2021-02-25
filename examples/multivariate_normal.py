"""
Example taken from Using Neural Networks to Model Conditional Multivariate Densities
Peter Martin Williams 1996
Replication of Figure 3.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal


def simulate_data(N=3000):
    x = np.random.rand(N) * np.pi
    x = np.sort(x)
    means = np.zeros((N, 2))
    means[:, 0] = np.sin(2.5 * x) * np.sin(1.5 * x)
    means[:, 1] = np.cos(3.5 * x) * np.cos(0.5 * x)
    cov = np.zeros((N, 2, 2))
    cov[:, 0, 0] = 0.01 + 0.25 * (1 - np.sin(2.5 * x)) ** 2
    cov[:, 1, 1] = 0.01 + 0.25 * (1 - np.cos(2.5 * x)) ** 2
    corr = np.sin(2.5 * x) * np.cos(0.5 * x)
    off_diag = corr * np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    cov[:, 0, 1] = off_diag
    cov[:, 1, 0] = off_diag
    scipy_dists = [
        multivariate_normal(mean=means[i, :], cov=cov[i, :, :]) for i in range(N)
    ]
    rvs = np.array([dist.rvs(1) for dist in scipy_dists])
    return x, rvs, scipy_dists


def cov_to_sigma(cov_mat):
    """
    Parameters:
        cov_mat: Nx2x2 numpy array
    Returns:
        sigma: (N,2) numpy array containing the variances
        corr: (N,) numpy array the correlation [-1,1] extracted from cov_mat
    """

    sigma = np.sqrt(np.diagonal(cov_mat, axis1=1, axis2=2))
    corr = cov_mat[:, 0, 1] / (sigma[:, 0] * sigma[:, 1])
    return sigma, corr


if __name__ == "__main__":

    SEED = 12345
    np.random.seed(SEED)
    X, Y, true_dist = simulate_data()
    X = X.reshape(-1, 1)
    dist = MultivariateNormal(2)

    data_figure, data_axs = plt.subplots()
    data_axs.plot(X, Y[:, 0], label="Dim 1")
    data_axs.plot(X, Y[:, 1], label="Dim 2")
    data_axs.set_xlabel("X")
    data_axs.set_ylabel("Y")
    data_axs.set_title("Input Data")
    data_axs.legend()
    data_figure.show()

    X_val, Y_val, _ = simulate_data(500)
    X_val = X_val.reshape(-1, 1)
    ngb = NGBRegressor(
        Dist=dist, verbose=True, n_estimators=2000, natural_gradient=True
    )
    ngb.fit(X, Y, X_val=X_val, Y_val=Y_val, early_stopping_rounds=100)
    y_dist = ngb.pred_dist(X, max_iter=ngb.best_val_loss_itr)

    # Extract parameters for plotting
    mean = y_dist.mean()
    sigma, corrs = cov_to_sigma(y_dist.cov)
    true_cov_mat = np.array([dist.cov for dist in true_dist])
    true_mean = np.array([dist.mean for dist in true_dist])
    true_sigma, true_corrs = cov_to_sigma(true_cov_mat)

    # Plot the parameters in the sigma, correlation representation
    fig, axs = plt.subplots(5, 1, sharex=True)
    colors = ["blue", "red"]
    axs[4].set_xlabel("X")
    for i in range(2):
        axs[i].set_title("Mean Dimension:" + str(i))
        axs[i].plot(X, mean[:, i], label="fitted")
        axs[i].plot(X, true_mean[:, i], label="true")

        axs[2 + i].set_title("Marginal Standard Deviation Dimension: " + str(i))
        axs[2 + i].plot(X, sigma[:, i], label="fitted")
        axs[2 + i].plot(X, true_sigma[:, i], label="true")
    axs[4].set_title("Correlation")
    axs[4].plot(X, corrs, label="fitted")
    axs[4].plot(X, true_corrs, label="true")
    for i in range(5):
        axs[i].legend()
    fig.tight_layout()
    fig.show()
