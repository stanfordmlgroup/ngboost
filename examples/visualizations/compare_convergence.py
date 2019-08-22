import pickle
import itertools
import matplotlib as mpl
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from examples.loggers.loggers import RegressionLogger


if __name__ == "__main__":

    mpl.style.use("seaborn-dark-palette")
    plt.figure(figsize = (8, 3))

    for score, natural in itertools.product(("CRPS", "MLE"), (True, False)):

        outfile = open("results/regression/logs_%s_%s_%s_Normal.pkl" %
            ("simulated", score, natural), "rb")
        logger = pickle.load(outfile)

        natural_str = "NGD" if natural else "GD"
        if score == "MLE":
            plt.subplot(1, 2, 1)
            plt.plot(logger.nlls[1:],
                     linestyle = "-" if natural else "--",
                     label = score + "-" + natural_str,
                     color = "black")

        if score == "CRPS":
            plt.subplot(1, 2, 2)
            plt.plot(logger.crps[1:],
                     linestyle = "-" if natural else "--",
                     label = score + "-" + natural_str,
                     color = "black")

    plt.subplot(1, 2, 1)
    plt.xlabel("Iterations")
    plt.ylabel("Test NLL")
    plt.legend(fontsize=10)
    plt.subplot(1, 2, 2)
    plt.legend(fontsize=10)
    plt.xlabel("Iterations")
    plt.ylabel("Test CRPS")
    plt.tight_layout()
    plt.savefig("./figures/compare_convergence.pdf")
    plt.show()
