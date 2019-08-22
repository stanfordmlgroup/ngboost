import numpy as np
import pandas as pd
import pickle
from ngboost.evaluation import *


class RegressionLogger(object):

    def __init__(self, args):
        self.args = args
        self.r2s = []
        self.mses = []
        self.nlls = []
        self.crps = []
        self.calib_slopes = []
        self.calib_errors = []

    def tick(self, forecast, y_test):
        pred, obs, slope, intercept = calibration_regression(forecast, y_test)
        self.r2s += [r2_score(y_test, forecast.loc)]
        self.mses += [mean_squared_error(y_test, forecast.loc)]
        self.nlls += [-np.diag(forecast.logpdf(y_test)).mean()]
        self.crps += [np.diag(forecast.crps(y_test)).mean()]
        self.calib_slopes.append(slope)
        self.calib_errors.append(calculate_calib_error(pred, obs))

    def to_row(self):
        self.calib_errors = np.array(self.calib_errors) * 100.0
        return pd.DataFrame({
            "dataset": [self.args.dataset],
            "distn": [self.args.distn],
            "score": [self.args.score],
            "rmse": [f"{np.mean(np.sqrt(self.mses)):.2f} $\pm$ {np.std(np.sqrt(self.mses)):.2f}"],
            "nll": [f"{np.mean(np.array(self.nlls)):.2f} $\pm$ {np.std(np.array(self.nlls)):.2f}"],
            "crps": [f"{np.mean(np.array(self.crps)):.2f} $\pm$ {np.std(np.array(self.crps)):.2f}"],
            "r2": [f"{np.mean(self.r2s):.2f} $\pm$ {np.std(self.r2s):.2f}"],
            "calib": [f"{np.mean(self.calib_errors):.2f} $\pm$ {np.std(self.calib_errors):.2f}"],
            "slope": [f"{np.mean(self.calib_slopes):.2f} $\pm$ {np.std(self.calib_slopes):.2f}"],
        })

    def print_results(self):
        print("R2: %.4f +/- %.4f" % (np.mean(self.r2s), np.std(self.r2s)))
        print("MSE: %.4f +/- %.4f" % (np.mean(self.mses), np.std(self.mses)))
        print("NLL: %.4f +/- %.4f" % (np.mean(self.nlls), np.std(self.nlls)))
        print("Calib: %.4f +/- %.4f" % (np.mean(self.calib_scores),
                                        np.std(self.calib_scores)))
        print("Slope: %.4f +/- %.4f" % (np.mean(self.calib_slopes),
                                        np.std(self.calib_slopes)))

    def save(self):
        outfile = open("results/regression/logs_%s_%s_%s_%s.pkl" %
            (self.args.dataset, self.args.score, self.args.natural,
             self.args.distn), "wb")
        pickle.dump(self, outfile)


class SurvivalLogger(object):

    def __init__(self, args):
        self.args = args
        self.r2s = []
        self.mses = []
        self.nlls = []
        self.crps = []
        self.calib_slopes = []
        self.calib_errors = []

    def tick(self, forecast, y_test):
        pred, obs, slope, intercept = calibration_regression(forecast, y_test)
        self.r2s += [r2_score(y_test, forecast.loc)]
        self.mses += [mean_squared_error(y_test, forecast.loc)]
        self.nlls += [-np.diag(forecast.logpdf(y_test)).mean()]
        self.crps += [np.diag(forecast.crps(y_test)).mean()]
        self.calib_slopes.append(slope)
        self.calib_errors.append(calculate_calib_error(pred, obs))

    def to_row(self):
        return pd.DataFrame({
            "dataset": [self.args.dataset],
            "score": [self.args.score],
            "natural": [self.args.natural],
            "rmse_mean": [np.mean(np.sqrt(self.mses))],
            "rmse_sd": [np.std(np.sqrt(self.mses))],
            "nll_mean": [np.mean(np.array(self.nlls))],
            "nll_sd": [np.std(np.array(self.nlls))],
            "crps_mean": [np.mean(np.array(self.crps))],
            "crps_sd": [np.std(np.array(self.crps))],
            "r2s_mean": [np.mean(self.r2s)],
            "r2s_sd": [np.std(self.r2s)],
            "calib_mean": [np.mean(self.calib_errors)],
            "calib_sd": [np.std(self.calib_errors)],
            "slope_mean": [np.mean(self.calib_slopes)],
            "slope_sd": [np.std(self.calib_slopes)],
        })

    def print_results(self):
        print("R2: %.4f +/- %.4f" % (np.mean(self.r2s), np.std(self.r2s)))
        print("MSE: %.4f +/- %.4f" % (np.mean(self.mses), np.std(self.mses)))
        print("NLL: %.4f +/- %.4f" % (np.mean(self.nlls), np.std(self.nlls)))
        print("Calib: %.4f +/- %.4f" % (np.mean(self.calib_scores),
                                        np.std(self.calib_scores)))
        print("Slope: %.4f +/- %.4f" % (np.mean(self.calib_slopes),
                                        np.std(self.calib_slopes)))

    def save(self):
        outfile = open("results/regression/logs_%s_%s_%s_%s.pkl" %
            (self.args.dataset, self.args.score, self.args.natural,
             self.args.distn), "wb")
        pickle.dump(self, outfile)