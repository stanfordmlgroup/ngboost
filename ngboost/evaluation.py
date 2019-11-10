import numpy as np
from tqdm import tqdm
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error


def calibration_regression(Forecast, Y, bins=11, eps=1e-3):
    """
    Calculate calibration in the regression setting.
    """
    pctles = np.linspace(eps, 1-eps, bins)
    observed = np.zeros_like(pctles)
    for i, pctle in enumerate(pctles):
        icdfs = Forecast.ppf(pctle)[:,np.newaxis]
        observed[i] = np.mean(Y < icdfs)
    slope, intercept = np.polyfit(pctles, observed, deg=1)
    return pctles, observed, slope, intercept


def calibration_time_to_event(Forecast, T, C, bins=10, eps=1e-3):
    """
    Calculate calibration in the time-to-event setting, using the probability
    integral transform and a Kaplan-Meier fit.

    Evaluate at percentiles of censoring.
    """
    cdfs = Forecast.cdf(T)
    kmf = KaplanMeierFitter()
    kmf.fit(cdfs, 1 - C)
    idxs = np.round(np.linspace(0, len(kmf.survival_function_) - 1, 11))
    preds = np.array(kmf.survival_function_.iloc[idxs].index)
    obs = 1 - np.array(kmf.survival_function_.iloc[idxs].KM_estimate)
    slope, intercept = np.polyfit(preds, obs, deg=1)
    return preds, obs, slope, intercept


def calculate_calib_error(predicted, observed):
    return np.sum((predicted - observed) ** 2) / len(predicted)


def plot_pit_histogram(predicted, observed, **kwargs):
    plt.bar(x = predicted[1:], height = np.diff(observed),
            width = -np.diff(predicted), align = "edge",
            fill = False, edgecolor = "black", **kwargs)
    plt.xlim((0, 1))
    plt.xlabel("Probability Integral Transform")
    plt.ylabel("Density")
    plt.axhline(1.0 / (len(predicted) - 1), linestyle = "--", color = "grey")
    plt.title("PIT Histogram")


def plot_calibration_curve(predicted, observed):
    """
    Plot calibration curve.
    """
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    plt.plot(predicted, observed, "o", color="black")
    plt.plot(np.linspace(0, 1), np.linspace(0, 1) * slope + intercept, "--",
             label="Slope: %.2f, Intercept: %.2f" % (slope, intercept),
             alpha=0.5, color="black")
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="grey",
             alpha=0.5)
    plt.xlabel("Predicted CDF")
    plt.ylabel("Observed CDF")
    plt.title("Calibration Plot")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc="upper left")
    # plt.show()


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
