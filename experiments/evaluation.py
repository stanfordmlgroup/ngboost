import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, r2_score


sns.set_style("ticks")
sns.set_palette(sns.color_palette("dark", 8))
plt_colors = sns.color_palette()
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"



def calibration_regression(Forecast, Y, bins=10, eps=1e-3):
    """
    Calculate calibration in the regression setting.

    Returns: list of predicted vs observed
    """
    pctles = np.linspace(eps, 1-eps, bins)
    observed = np.zeros_like(pctles)
    for i, pctle in enumerate(pctles):
        icdfs = Forecast.icdf(torch.tensor(pctle)).detach().numpy()
        observed[i] = np.mean(Y < icdfs)
    slope, intercept = np.polyfit(pctles, observed, deg=1)
    return pctles, observed, slope, intercept


def plot_calibration_curve(predicted, observed):

    slope, intercept = np.polyfit(predicted, observed, deg=1)
    plt.plot(predicted, observed, "o")
    plt.plot(np.linspace(0, 1), np.linspace(0, 1) * slope + intercept, "--",
             label="Slope: %.2f, Intercept: %.2f" % (slope, intercept),
             alpha=.5, color=plt_colors[0])
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="black",
             alpha=.5)
    plt.xlabel("Predicted CDF")
    plt.ylabel("Observed CDF")
    plt.title("Calibration plot")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc="upper left")
    plt.show()

def empirical_loglik():
    """
    Calculate the empirical log-likihood.
    """
    pass

def calculate_concordance_dead_only(preds, ys, cs):
    """
    Calculate C-statistic for only cases where outcome is uncensored.
    """
    return calculate_concordance_naive(np.array(preds[cs == 0]),
                                       np.array(ys[cs == 0]),
                                       np.array(cs[cs == 0]))


def calculate_concordance_naive(preds, ys, cs):
    """
    Calculate Harrell's C-statistic in the presence of censoring.

    Cases:
    - (c=0, c=0): both uncensored, can compare
    - (c=0, c=1): can compare if true censored time > true uncensored time
    - (c=1, c=0): can compare if true censored time > true uncensored time
    - (c=1, c=1): both censored, cannot compare
    """
    trues = ys
    concordance, N = 0, len(trues)
    counter = 0
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            if (not cs[i] and not cs[j]) or \
                 (not cs[i] and cs[j] and ys[i] < ys[j]) or \
                 (cs[i] and not cs[j] and ys[i] > ys[j]):
                if (preds[i] < preds[j] and trues[i] < trues[j]) or \
                     (preds[i] > preds[j] and trues[i] > trues[j]):
                        concordance += 1
                elif preds[i] == preds[j]:
                    concordance += 0.5
                counter += 1
    return concordance / counter
