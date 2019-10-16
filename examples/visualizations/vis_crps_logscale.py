import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import itertools
from ngboost.distns import Normal, LogNormal
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == "__main__":

    np.random.seed(1)
    rvs = np.exp(np.random.normal(500,)

    logscale_axis = np.linspace(-3, 3, 1000)
    lognorm_axis = np.linspace(1e-8, 5, 1000)
    logscale_cdf = Normal(np.array([0, 0]), temp_scale = 1.0).cdf(logscale_axis)
    lognorm_cdf = LogNormal(np.array([0, 1]), temp_scale = 1.0).cdf(lognorm_axis)

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 2)
    plt.xlabel("$\\log x$")
    plt.ylabel("$F_\\theta(\\log x)$")
    plt.plot(logscale_axis, logscale_cdf, color = "black")
    plt.fill_between(logscale_axis, logscale_cdf,
                     where = logscale_axis < 0, color = "grey")
    plt.fill_between(logscale_axis, 1, logscale_cdf,
                     where = logscale_axis > 0, color = "grey")
    plt.axvline(0, color = "grey")
    plt.title("Log-scale")
    plt.subplot(1, 2, 1)
    plt.plot(lognorm_axis, lognorm_cdf, color = "black")
    plt.axvline(1, color = "grey")
    plt.xlabel("$x$")
    plt.ylabel("$F_\\theta(x)$")
    plt.title("Original scale")
    plt.fill_between(lognorm_axis, lognorm_cdf,
                     where = lognorm_axis < 1, color = "grey")
    plt.fill_between(lognorm_axis, 1, lognorm_cdf,
                     where = lognorm_axis > 1, color = "grey")
    plt.tight_layout()
    plt.savefig("./figures/crps_logscale.pdf")
    plt.show()

    logscale_crps_fn = lambda p: Normal(p, temp_scale = 1.0).crps(np.log(rvs)).mean()
    lognorm_crps_fn = lambda p: LogNormal(p, temp_scale = 1.0).crps(rvs).mean()
    logscale_crps_grad_fn = grad(logscale_crps_fn)
    lognorm_crps_grad_fn = grad(lognorm_crps_fn)

    loc = np.linspace(-1, 1, 20)
    scale = np.linspace(-0.5, 1, 20)

    loc, scale = np.meshgrid(loc, scale)

    grads_logscale_x = np.zeros((20, 20))
    grads_logscale_y = np.zeros((20, 20))
    grads_lognorm_x = np.zeros((20, 20))
    grads_lognorm_y = np.zeros((20, 20))
    logscale_crps = np.zeros((20, 20))
    lognorm_crps = np.zeros((20, 20))

    for (i, j) in tqdm(itertools.product(np.arange(20), np.arange(20))):
        #H = np.linalg.inv(np.array(hessian_fn([loc[i, j], scale[i, j]])))
        # H = np.linalg.inv(metric_fn([loc[i, j], scale[i, j]]))
        g = np.array(logscale_crps_grad_fn([loc[i, j], scale[i, j]]))
        grads_logscale_x[i, j] = -g[0]
        grads_logscale_y[i, j] = -g[1]
        g = np.array(lognorm_crps_grad_fn([loc[i, j], scale[i, j]]))
        grads_lognorm_x[i, j] = -g[0]
        grads_lognorm_y[i, j] = -g[1]
        logscale_crps[i, j] = logscale_crps_fn([loc[i, j], scale[i, j]])
        lognorm_crps[i, j] = lognorm_crps_fn([loc[i, j], scale[i, j]])

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 2)
    plt.contourf(loc, scale, logscale_crps, cmap = mpl.cm.viridis, levels = 100)
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma^2$")
    plt.title("CRPS: log-scale")
    plt.subplot(1, 2, 1)
    plt.contourf(loc, scale, lognorm_crps, cmap = mpl.cm.viridis, levels = 100)
    plt.title("CRPS: original scale")
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma^2$")
    plt.tight_layout()
    plt.savefig("./figures/vis_crps_logscale.pdf")
    plt.show()
