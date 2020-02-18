from ngboost.distns import ClassificationDistn
from ngboost.scores import LogScore, CRPScore
import numpy as np
import scipy as sp
import scipy.special


class CategoricalLogScore(LogScore):
    def score(self, Y):
        return -np.log(self.probs[Y, range(len(Y))])

    def d_score(self, Y):
        return (self.probs.T - np.eye(self.K_)[Y])[:, 1 : self.K_]

    def metric(self):
        FI = -np.einsum(
            "ji,ki->ijk", self.probs[1 : self.K_, :], self.probs[1 : self.K_, :]
        )
        d = np.einsum("jii->ij", FI)
        d[:] += self.probs[1 : self.K_, :]
        return FI

    # a test:
    # if k==j:
    #     a= FI[i,j,k] == self.probs[k,i] - self.probs[k,i]*self.probs[j,i]
    # else:
    #     a= FI[i,j,k] == -self.probs[k,i]*self.probs[j,i]
    # a


class CategoricalCRPScore(CRPScore):
    def score(self, Y):
        return np.sum((self.probs - np.eye(self.K_)[Y]) ** 2, axis=1)

    def d_score(self, Y):
        return None

    def metric(self):
        return None


def k_categorical(K):
    """
    Factory function that generates classes for K-class categorical distributions for NGBoost

    The generated distribution has two parameters, loc and scale, which are the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    class Categorical(ClassificationDistn):

        scores = [CategoricalLogScore]
        problem_type = "classification"
        n_params = K - 1
        K_ = K

        def __init__(self, params):
            super().__init__(params)
            _, N = params.shape
            self.logits = np.zeros((K, N))
            self.logits[1:K, :] = params  # default the 0th class logits to 0
            self.probs = sp.special.softmax(self.logits, axis=0)
            # self.dist = dist(n=1, p=self.probs) # scipy doesn't allow vectorized multinomial (!?!?) why allow vectorized versions of the others?
            # this makes me want to refactor all the other code to use lists of distributions, would be more readable imo

        def fit(Y):
            _, n = np.unique(Y, return_counts=True)
            p = n / len(Y)
            return np.log(p[1:K]) - np.log(p[0])
            # https://math.stackexchange.com/questions/2786600/invert-the-softmax-function

        def sample1(self):  # this is just a helper for sample()
            cum_p = np.cumsum(self.probs, axis=0)[0:-1]
            interval = cum_p < np.random.random((1, len(self)))
            return np.sum(interval, axis=0)

        def sample(self, m):
            return np.array([self.sample1() for i in range(m)])

        def class_probs(self):  # required for any ClassificationDistn
            return self.probs.T

        @property
        def params(self):
            names = [f"p{j}" for j in range(self.n_params + 1)]
            return {name: p for name, p in zip(names, self.probs)}

    return Categorical


Bernoulli = k_categorical(2)
