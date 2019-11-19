import numpy as np
import pandas as pd
import pickle
from ngboost.evaluation import *


class RegressionLogger(object):

    def __init__(self, args):
        self.args = args
        self.log = pd.DataFrame()

    def tick(self, forecast, y_test):
        pred, obs, slope, intercept = calibration_regression(forecast, y_test)
        self.log = self.log.append([{
            "r2": r2_score(y_test, forecast.loc),
            "mse": mean_squared_error(y_test, forecast.loc),
            "nll": -forecast.logpdf(y_test.flatten()).mean(),
            "crps": forecast.crps(y_test.flatten()).mean(),
            "slope": slope,
            "calib": calculate_calib_error(pred, obs)
        }])

    def to_row(self):
        return pd.DataFrame({
            "dataset": [self.args.dataset],
            "distn": [self.args.distn],
            "score": [self.args.score],
            "rmse": ["{:.2f} \pm {:.2f}".format(np.mean(np.sqrt(self.log["mse"])), np.std(np.sqrt(self.log["mse"])) / self.args.reps)],
            "nll": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["nll"]), np.std(self.log["nll"]) / self.args.reps)],
            "crps": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["crps"]), np.std(self.log["crps"]) / self.args.reps)],
            "r2": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["r2"]), np.std(self.log["r2"]) / self.args.reps)],
            "calib": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["calib"]), np.std(self.log["calib"]) / self.args.reps)],
            "slope": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["slope"]), np.std(self.log["slope"]) / self.args.reps)],
        })

    def save(self):
        outfile = open("results/regression/logs_%s_%s_%s_%s.pkl" %
            (self.args.dataset, self.args.score, self.args.natural,
             self.args.distn), "wb")
        pickle.dump(self, outfile)


class SurvivalLogger(object):

    def __init__(self, args):
        self.args = args
        self.log = pd.DataFrame()

    def tick(self, forecast, y_test):
        C = 1-y_test['Event']
        T = y_test['Time']
        pred, obs, slope, intercept = calibration_time_to_event(forecast, T, C)
        self.log = self.log.append([{
            "cstat_naive": calculate_concordance_naive(forecast.loc, T, C),
            "cstat_dead": calculate_concordance_dead_only(forecast.loc, T, C),
            "cov": np.mean(np.sqrt(forecast.var) / forecast.loc),
            "slope": slope,
            "calib": calculate_calib_error(pred, obs)
        }])

    def to_row(self):
        return pd.DataFrame({
            "dataset": [self.args.dataset],
            "score": [self.args.score],
            "natural": [self.args.natural],
            "cstat_naive": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["cstat_naive"]), np.std(self.log["cstat_naive"]) / self.args.reps)],
            "cstat_dead": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["cstat_dead"]), np.std(self.log["cstat_dead"]) / self.args.reps)],
            "calib": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["calib"]), np.std(self.log["calib"]) / self.args.reps)],
            "slope": ["{:.2f} \pm {:.2f}".format(np.mean(self.log["slope"]), np.std(self.log["slope"]) / self.args.reps)],
        })

    def save(self):
        outfile = open("results/survival/logs_%s_%s_%s_%s.pkl" %
            (self.args.dataset, self.args.score, self.args.natural,
             self.args.distn), "wb")
        pickle.dump(self, outfile)
