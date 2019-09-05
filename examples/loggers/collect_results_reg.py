import pandas as pd
import pickle
import itertools
from examples.loggers.loggers import RegressionLogger


if __name__ == "__main__":

    rows = []
    datasets = ("concrete", "energy", "housing", "wine", "yacht")
    dists = ("Normal", "Laplace")

    for (dataset, dist) in itertools.product(datasets, dists):

        name = f"logs_{dataset}_CRPS_False_{dist}"
        logs = pickle.load(open(f"./results/regression/{name}.pkl", "rb"))
        rows += [logs.to_row()]

    print("===")
    print(pd.concat(rows).to_latex(index = False, float_format = "%.2f",
                                   escape =  False))
    print("===")
