import numpy as np
from ngboost.distns import Normal, Laplace
from ngboost.evaluation import *
from matplotlib import pyplot as plt


if __name__ == "__main__":

    axis = np.linspace(-4, 4, 1000)
    d1 = Normal(np.array([0., 0.]))
    d2 = Laplace(np.array([0., 0.]))
    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(axis, np.exp(d1.logpdf(axis)), color = "black", label = "Gaussian")
    plt.plot(axis, np.exp(d2.logpdf(axis)), color = "grey", label = "Laplace")
    plt.xlabel("$x$")
    plt.ylabel("$p_{\\theta}(x)$")
    plt.legend(fontsize=10)
    plt.title("Distribution Comparison")
    plt.subplot(1, 2, 2)
    plot_pit_histogram([0.001, 0.1008, 0.2006, 0.3004, 0.4002, 0.5, 0.5998, 0.6996, 0.7994, 0.8992, 0.999 ],
                       [0.002, 0.095, 0.19,  0.30, 0.402, 0.503, 0.598, 0.693, 0.804, 0.903, 0.999])
    plt.tight_layout()
    plt.savefig("./figures/pit_correct.pdf")
    plt.show()
