from ngboost.distns import Distn
import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import multinomial as dist

def k_categorical(K):
    class Categorical(Distn):

        problem_type = "classification"
        n_params = K-1

        def __init__(self, params):
            _, N = params.shape
            self.params_ = params
            self.logits = np.zeros((K, N))
            self.logits[1:K,:] = params # default the 0th class logits to 0
            self.probs = sp.special.softmax(self.logits, axis=0)
            # self.dist = dist(n=1, p=self.probs) # scipy doesn't allow vectorized multinomial (!?!?) why allow vectorized versions of the others?
            # this makes me want to refactor all the other code to use lists of distributions, would be more readable imo

        @property
        def params(self):
            names = [f'p{j}' for j in range(self.n_params+1)]
            return {name:p for name, p in zip(names, self.probs)}

        def to_prob(self):
            return self.probs.T

        def fit(Y):
            _, n = np.unique(Y, return_counts=True)
            p = n/len(Y)
            return np.log(p[1:K]) - np.log(p[0])
            # https://math.stackexchange.com/questions/2786600/invert-the-softmax-function

        def sample1(self):
            cum_p = np.cumsum(self.probs, axis=0)[0:-1]
            interval = cum_p < np.random.random((1,len(self)))
            return np.sum(interval, axis=0)            

        def sample(self, m):
            return np.array([self.sample1() for i in range(m)])

        # log score methods
        def nll(self, Y):
            return -np.log(self.probs[Y, range(len(Y))])

        def D_nll(self, Y):
            return (self.probs.T - np.eye(K)[Y])[:,1:K]

        def fisher_info(self):
            FI = -np.einsum('ji,ki->ijk', self.probs[1:K,:], self.probs[1:K,:])
            d = np.einsum('jii->ij', FI)
            d[:] += self.probs[1:K,:]
            return FI

            # a test:
            # if k==j:
            #     a= FI[i,j,k] == self.probs[k,i] - self.probs[k,i]*self.probs[j,i]
            # else:
            #     a= FI[i,j,k] == -self.probs[k,i]*self.probs[j,i]
            # a

        # crps methods
        def crps(self, Y):
            return np.sum((self.probs - np.eye(K)[Y])**2, axis=1)

        # def D_crps(self, Y):

        # def crps_metric(self):
        #     M = 2 * self.prob ** 2 * np.exp(-2 * self.logit) * (1 + (self.prob / (1 - self.prob)) ** 2)
        #     return M[:, np.newaxis, np.newaxis]

    return Categorical

Bernoulli = k_categorical(2)
