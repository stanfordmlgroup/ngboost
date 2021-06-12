"""The NGBoost mixture of K Normal distributions and scores"""

import scipy
from scipy.stats import norm
from scipy.stats import laplace as dist
import numpy as np
import math as math
import pandas as pd
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from sklearn.cluster import KMeans



class NormalMixtureLogScore(LogScore):
    def score(self, Y):
        return -np.log(np.sum(norm.pdf(Y, self.loc, self.scale)*self.mixprop, axis = 0))


    def d_score(self, Y):
        K = self.K_
        D = np.zeros((len(Y), (3 * K - 1)))

        D[:, range(K)] = np.transpose(-1/(np.sum(norm.pdf(Y, self.loc, self.scale)*self.mixprop, axis = 0))*self.mixprop*((Y - self.loc) / pow(self.scale, 2))*norm.pdf(Y, self.loc, self.scale))
        
        D[:, range(K, (2 * K))] = np.transpose(-1/(np.sum(norm.pdf(Y, self.loc, self.scale)*self.mixprop, axis = 0))*self.mixprop*((pow((Y - self.loc), 2) - pow(self.scale, 2)) / pow(self.scale, 2))*norm.pdf(Y, self.loc, self.scale))
        
        D_alpha = np.transpose(-1/(np.sum(norm.pdf(Y, self.loc, self.scale)*self.mixprop, axis = 0))*(norm.pdf(Y, self.loc, self.scale)[range(K-1)] - norm.pdf(Y, self.loc, self.scale)[K-1]))
                    
        m = np.einsum("ij, kj -> jik", self.mixprop[range(K-1)], mixprop[range(K-1)])
        d = np.einsum("ijj -> ij", m)
        d -= np.einsum("i...", self.mixprop[range(K-1)])

        D[:, range(2 * K, (3 * K - 1))] = np.einsum("ij, ijl -> il", D_alpha, -m)
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
            self.mixprop = exp_mixprop/np.sum(exp_mixprop, axis=0)

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
                    np.random.multinomial(n=1, pvals=self.mixprop[:, i], size=m)
                    for i in range(self.mixprop.shape[1])
                ]
            ).transpose(1, 2, 0)
            samples = norm.rvs(self.loc, self.scale, size=(m,) + self.loc.shape)
            return np.sum(component * samples, axis=1)

        def mean(self,):
            n = self._params.shape[1]
            np.sum(self.mixprop * self.loc, axis=0)

        @property
        def params(self):
            return {"loc": self.loc, "scale": self.scale, "mix_prop": self.mixprop}

    return NormalMixture

