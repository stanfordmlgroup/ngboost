import pandas as pd
import pickle
import itertools
from examples.loggers.loggers import RegressionLogger


if __name__ == "__main__":

    rows = []
    datasets = ("flchain",)
    dists = ("Normal",)

    for (dataset, dist) in itertools.product(datasets, dists):

        name = f"logs_{dataset}_MLE_Surv_False_{dist}"
        logs = pickle.load(open(f"./results/survival/{name}.pkl", "rb"))
        rows += [logs.to_row()]

    print("===")
    print(pd.concat(rows).to_latex(index=False, float_format="%.2f", escape=False))
    print("===")
