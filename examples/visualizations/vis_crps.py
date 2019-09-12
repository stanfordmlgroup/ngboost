import numpy as onp
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats
import jax.random as random
import matplotlib as mpl
import itertools
from ngboost.distns import Normal, Laplace
from jax import grad, vmap, jacrev, jit
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == "__main__":

    key = random.PRNGKey(seed=123)
    rvs = random.normal(key=key, shape=(500,))

    crps_fn = lambda p: Normal(p, temp_scale=1.0).crps(rvs).mean()
    metric_fn = lambda p: Normal(p, temp_scale=1.0).crps_metric()
    grad_fn = grad(crps_fn)
    hessian_fn = jacrev(grad_fn)

    loc = onp.linspace(-3, 3, 20)
    scale = onp.linspace(-0.5, 2, 20)

    loc, scale = onp.meshgrid(loc, scale)

    grads_metric_x = onp.zeros((20, 20))
    grads_metric_y = onp.zeros((20, 20))
    grads_x = onp.zeros((20, 20))
    grads_y = onp.zeros((20, 20))
    crps = onp.zeros((20, 20))

    for (i, j) in tqdm(itertools.product(np.arange(20), np.arange(20))):
        #H = np.linalg.inv(np.array(hessian_fn([loc[i, j], scale[i, j]])))
        H = np.linalg.inv(metric_fn([loc[i, j], scale[i, j]]))
        g = np.array(grad_fn([loc[i, j], scale[i, j]]))
        gf = H @ g
        grads_metric_x[i, j] = -gf[0]
        grads_metric_y[i, j] = -gf[1]
        grads_x[i, j] = -g[0]
        grads_y[i, j] = -g[1]
        crps[i, j] = crps_fn([loc[i, j], scale[i, j]])

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plt.contourf(loc, scale, crps, cmap = mpl.cm.viridis, levels = 100)
    plt.quiver(loc, scale, 0.14 * grads_x, 0.14 * grads_y,
               color = "white", angles='xy', scale_units='xy', scale=1)
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma$")
    plt.title("CRPS: gradients")
    plt.subplot(1, 2, 2)
    plt.contourf(loc, scale, crps, cmap = mpl.cm.viridis, levels = 100)
    plt.quiver(loc, scale, 0.07 * grads_metric_x, 0.07 * grads_metric_y,
               color = "white", angles='xy', scale_units='xy', scale=1)
    plt.title("CRPS: natural gradients")
    plt.xlabel("$\mu$")
    plt.ylabel("$\log\sigma$")
    plt.tight_layout()
    plt.savefig("./figures/vis_crps.pdf")
    plt.show()
