import numpy as np
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from tqdm import tqdm


def calibration_regression(Forecast, Y, bins=11, eps=1e-3):
    """
    Calculate calibration in the regression setting.
    """
    pctles = np.linspace(eps, 1 - eps, bins)
    observed = np.zeros_like(pctles)
    for i, pctle in enumerate(pctles):
        icdfs = Forecast.ppf(pctle).reshape(Y.shape)
        observed[i] = np.mean(Y < icdfs)
    slope, intercept = np.polyfit(pctles, observed, deg=1)
    return pctles, observed, slope, intercept


def calibration_time_to_event(Forecast, T, E):
    """
    Calculate calibration in the time-to-event setting, with integral transform and KM.
    """
    cdfs = Forecast.cdf(T)
    kmf = KaplanMeierFitter()
    kmf.fit(cdfs, E)
    idxs = np.round(np.linspace(0, len(kmf.survival_function_) - 1, 11))
    preds = np.array(kmf.survival_function_.iloc[idxs].index)
    obs = 1 - np.array(kmf.survival_function_.iloc[idxs].KM_estimate)
    slope, intercept = np.polyfit(preds, obs, deg=1)
    return preds, obs, slope, intercept


def calculate_calib_error(predicted, observed):
    return np.sum((predicted - observed) ** 2) / len(predicted)


def plot_pit_histogram(predicted, observed, **kwargs):
    plt.bar(
        x=predicted[1:],
        height=np.diff(observed),
        width=-np.diff(predicted),
        align="edge",
        fill=False,
        edgecolor="black",
        **kwargs
    )
    plt.xlim((0, 1))
    plt.xlabel("Probability Integral Transform")
    plt.ylabel("Density")
    plt.axhline(1.0 / (len(predicted) - 1), linestyle="--", color="grey")
    plt.title("PIT Histogram")


def plot_calibration_curve(predicted, observed):
    """
    Plot calibration curve.
    """
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    plt.plot(predicted, observed, "o", color="black")
    plt.plot(
        np.linspace(0, 1),
        np.linspace(0, 1) * slope + intercept,
        "--",
        label="Slope: %.2f, Intercept: %.2f" % (slope, intercept),
        alpha=0.5,
        color="black",
    )
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="grey", alpha=0.5)
    plt.xlabel("Predicted CDF")
    plt.ylabel("Observed CDF")
    plt.title("Calibration Plot")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc="upper left")


def calculate_concordance_dead_only(preds, Y, E):
    """
    Calculate C-statistic for only cases where outcome is uncensored.
    """
    return calculate_concordance_naive(
        np.array(preds[E == 1]), np.array(Y[E == 1]), np.array(E[E == 1])
    )


def calculate_concordance_naive(preds, Y, E):
    """
    Calculate Harrell's C-statistic in the presence of censoring.

    Cases:
    - (c=0, c=0): both uncensored, can compare
    - (c=0, c=1): can compare if true censored time > true uncensored time
    - (c=1, c=0): can compare if true censored time > true uncensored time
    - (c=1, c=1): both censored, cannot compare
    """
    trues = Y
    concordance, N = 0, len(trues)
    counter = 0
    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            cond_1 = E[i] and E[j]
            cond_2 = E[i] and not E[j] and Y[i] < Y[j]
            cond_3 = not E[i] and E[j] and Y[i] > Y[j]
            if cond_1 or cond_2 or cond_3:
                if (preds[i] < preds[j] and trues[i] < trues[j]) or (
                    preds[i] > preds[j] and trues[i] > trues[j]
                ):
                    concordance += 1
                elif preds[i] == preds[j]:
                    concordance += 0.5
                counter += 1
    return concordance / counter
