import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import multinomial as dist

def k_categorical(K):
    class Categorical(object):

        problem_type = "classification"
        n_params = K-1

        def __init__(self, params):
            _, N = params.shape
            self.logits = np.zeros((K, N))
            self.logits[1:K,:] = params # default the 0th class logits to 0
            self.probs = sp.special.softmax(self.logits, axis=0)
            # self.dist = dist(n=1, p=self.probs) # scipy doesn't allow vectorized multinomial (!?!?) why allow vectorized versions of the others?
            # this makes me want to refactor all the other code to use lists of distributions, would be more readable imo

        # def __getattr__(self, name):
        #     if name in dir(self.dist):
        #         return getattr(self.dist, name)
        #     return None

        def to_prob(self): 
            return self.probs.T

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

        # def crps(self, Y):
        #     return ((self.prob - Y) ** 2)

        # def D_crps(self, Y):
        #     D = 2 * (self.prob - Y) * self.prob ** 2 * np.exp(-self.logit)
        #     return D[:,np.newaxis]

        # def crps_metric(self):
        #     M = 2 * self.prob ** 2 * np.exp(-2 * self.logit) * (1 + (self.prob / (1 - self.prob)) ** 2)
        #     return M[:, np.newaxis, np.newaxis]

        def fit(Y):
            _, n = np.unique(Y, return_counts=True)
            p = n/len(Y)
            return np.log(p[1:K]) - np.log(p[0]) 
            # https://math.stackexchange.com/questions/2786600/invert-the-softmax-function

    return Categorical