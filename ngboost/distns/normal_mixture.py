from scipy.stats import norm
import numpy as np
import math as math
import pandas as pd
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from sklearn.cluster import KMeans


class NormalMixtureLogScore(LogScore):
    def score(self, Y):
        K = self.K_
        n = len(Y)
        return -np.log(
            [
                np.sum(
                    self.mixprop[:, j]
                    * [
                        norm.pdf(Y[j], self.loc[i, j], self.scale[i, j])
                        for i in range(K)
                    ]
                )
                for j in range(n)
            ]
        )

    def inv_f(self, z, j):  # helper
        return 1 / np.sum(
            self.mixprop[:, j]
            * [norm.pdf(z, self.loc[i, j], self.scale[i, j]) for i in range(self.K_)]
        )

    def d_score(self, Y):
        K = self.K_
        n = len(Y)
        D = np.zeros((len(Y), (3 * K - 1)))

        D[:, range(K)] = [
            -self.inv_f(Y[j], j)
            * self.mixprop[:, j]
            * (1 / np.sqrt(2 * np.pi))
            * ((Y[j] - self.loc[:, j]) / pow(self.scale[:, j], 3))
            * np.exp(
                -0.5 * (pow((Y[j] - self.loc[:, j]), 2) / pow(self.scale[:, j], 2))
            )
            for j in range(n)
        ]
        D[:, range(K, (2 * K))] = [
            -self.inv_f(Y[j], j)
            * self.mixprop[:, j]
            * (1 / np.sqrt(2 * np.pi))
            * (
                (pow((Y[j] - self.loc[:, j]), 2) - pow(self.scale[:, j], 2))
                / pow(self.scale[:, j], 3)
            )
            * np.exp(
                -0.5 * (pow((Y[j] - self.loc[:, j]), 2) / pow(self.scale[:, j], 2))
            )
            for j in range(n)
        ]
        D_alpha = np.array(
            [
                [
                    self.inv_f(Y[j], j)
                    * (
                        norm.pdf(Y[j], self.loc[K - 1, j], self.scale[K - 1, j])
                        - norm.pdf(Y[j], self.loc[i, j], self.scale[i, j])
                    )
                    for i in range(K - 1)
                ]
                for j in range(n)
            ]
        )
        D_alpha1 = np.zeros((len(Y), (K - 1)))
        for i in range(n):
            if K == 2:
                inv_Jaccobian = self.mixprop[0, i] * (1 - self.mixprop[0, i])
                D_alpha1 = D_alpha * inv_Jaccobian
            else:
                inv_Jaccobian = np.linalg.inv(
                    np.diag(1 / self.mixprop[range(K - 1), i])
                    + (1 / self.mixprop[K - 1, i]) * np.ones([K - 1, K - 1])
                )
                D_alpha1[i, :] = D_alpha[i, :] @ inv_Jaccobian

        D[:, range(2 * K, (3 * K - 1))] = D_alpha1
        return D


def k_normal_mixture(K):
    class NormalMixture(RegressionDistn):

        K_ = K
        n_params = 3 * K - 1
        scores = [NormalMixtureLogScore]

        def __init__(self, params):

            # save the parameters
            self._params = params

            # create other objects that will be useful later
            self.loc = params[0:K]
            self.logscale = params[K : (2 * K)]
            self.scale = np.exp(self.logscale)

            mix_params = np.zeros((K, params.shape[1]))
            mix_params[0 : (K - 1), :] = params[(2 * K) : (3 * K - 1)]
            exp_mixprop = np.exp(mix_params)
            self.mixprop = exp_mixprop / np.sum(exp_mixprop, axis=0)

        def fit(Y):
            kmeans = KMeans(n_clusters=K).fit(Y.reshape(-1, 1))
            pred = kmeans.predict(Y.reshape(-1, 1))
            loc = []
            scale = []
            prop = []
            for i in range(K):
                obs = Y[pred == i]
                loc = np.append(loc, np.mean(obs))
                scale = np.append(scale, np.std(obs))
                prop = np.append(prop, len(obs) / len(Y))
            return np.concatenate(
                [
                    loc,
                    np.log(scale),
                    np.log(prop[range(K - 1)] / (1 - sum(prop[range(K - 1)]))),
                ]
            )

        def sample(self, m):
            component = np.array(
                [  # it's stupid that there is no fast vectorized multinomial in python
                    np.random.multinomial(n=1, pvals=self.mixprop[:, i], size=10)
                    for i in range(self.mixprop.shape[1])
                ]
            ).transpose(1, 2, 0)
            samples = norm.rvs(self.loc, self.scale, size=(10,) + self.loc.shape)
            return np.sum(component * samples, axis=1)

        def mean(self,):
            n = self._params.shape[1]
            np.sum(self.mixprop * self.loc, axis=0)

        @property
        def params(self):
            return {"loc": self.loc, "scale": self.scale, "mix_prop": self.mixprop}

    return NormalMixture
