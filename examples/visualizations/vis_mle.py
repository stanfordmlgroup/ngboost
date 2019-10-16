import numpy as np
import scipy as sp
import scipy.stats
import np.random as random
import matplotlib as mpl
import itertools
from ngboost.distns import Normal, Laplace
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == "__main__":


    key = random.PRNGKey(seed=123)
    rvs = random.normal(key=key, shape=(500,))

    nll_fn = lambda p: -Normal(p, temp_scale = 1.0).logpdf(rvs).mean()
    fisher_fn = lambda p: Normal(p, temp_scale = 1.0).fisher_info()

    grad_fn = jit(grad(nll_fn))
    hessian_fn = jit(jacrev(grad_fn))

    loc = np.linspace(-3, 3, 20)
    scale = np.linspace(-0.5, 2, 20)
    loc, scale = np.meshgrid(loc, scale)

    grads_fisher_x = np.zeros((20, 20))
    grads_fisher_y = np.zeros((20, 20))
    grads_x = np.zeros((20, 20))
    grads_y = np.zeros((20, 20))
    nlls = np.zeros((20, 20))

    for (i, j) in tqdm(itertools.product(np.arange(20), np.arange(20))):
        #H = np.linalg.inv(np.array(hessian_fn([loc[i, j], scale[i, j]])))
        H = np.linalg.inv(fisher_fn([loc[i, j], scale[i, j]]))
        g = np.array(grad_fn([loc[i, j], scale[i, j]]))
        gf = H @ g
        grads_fisher_x[i, j] = -gf[0]
        grads_fisher_y[i, j] = -gf[1]
        grads_x[i, j] = -g[0]
        grads_y[i, j] = -g[1]
        nlls[i, j] = nll_fn([loc[i, j], scale[i, j]])

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plt.contourf(loc, scale, nlls, cmap = mpl.cm.viridis, levels = 100)
    plt.quiver(loc, scale, 0.07 * grads_x, 0.07 * grads_y,
               color = "white", angles='xy', scale_units='xy', scale=1)
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma$")
    plt.title("MLE: gradients")
    plt.subplot(1, 2, 2)
    plt.contourf(loc, scale, nlls, cmap = mpl.cm.viridis, levels = 100)
    plt.quiver(loc, scale, 0.07 * grads_fisher_x, 0.07 * grads_fisher_y,
               color = "white", angles='xy', scale_units='xy', scale=1)
    plt.title("MLE: natural gradients")
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma$")
    plt.tight_layout()
    plt.savefig("./figures/vis_mle.pdf")
    plt.show()
