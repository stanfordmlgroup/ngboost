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

    def inv_f(z, j):  # helper
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

    def metric(self):
        grads = np.stack([self.d_score(Y) for Y in self.sample(10000)])
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)


def k_NormalMixture(K):
    class NormalMixture(RegressionDistn):

        K_ = K
        n_params = 3 * K - 1
        scores = [NormalMixtureLogScore]

        def __init__(self, params):

            # save the parameters
            self._params = params

            # create other objects that will be useful later
            self.loc = params[range(K)]

            self.logscale = params[range(K, (2 * K))]
            self.scale = np.exp(params[range(K, (2 * K))])

            n = params.shape[1]
            self.transformed_mixprop = params[range(2 * K, (3 * K - 1))]
            mixprop = np.transpose(params[range(2 * K, (3 * K - 1))])
            mixprop1 = []
            for i in range(mixprop.shape[0]):
                prop = np.exp(mixprop[i]) / (1 + np.sum(np.exp(mixprop[i])))
                mixprop1 = np.append(mixprop1, np.append(prop, 1 - np.sum(prop)))
            self.mixprop = np.transpose(mixprop1.reshape(n, K))

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
            n = self._params.shape[1]
            final_data = []
            for j in range(n):
                cum_size = np.append(
                    0, [math.floor(i) for i in np.cumsum(self.mixprop[:, j]) * m]
                )
                data = np.array([])
                for i in range(K):
                    data = np.append(
                        data,
                        norm.rvs(
                            self.loc[i, j],
                            self.scale[i, j],
                            (cum_size[i + 1] - cum_size[i]),
                        ),
                    )
                final_data = np.append(final_data, data)
            return np.transpose(final_data.reshape(n, m))

        def mean(
            self,
        ):  # gives us access to Laplace.mean() required for RegressionDist.predict()
            n = self._params.shape[1]
            return [np.sum(self.mixprop[:, i] * self.loc[:, i]) for i in range(n)]

        @property
        def params(self):
            return {"loc": self.loc, "scale": self.scale, "mix_prop": self.mixprop}

    return NormalMixture
