import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import multinomial as dist

def k_categorical(K):
    class Categorical(object):

        problem_type = "classification"
        n_params = K

        def __init__(self, params):
            self.logits = params
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
            return self.probs.T - np.eye(Y.max() + 1)[Y]

        def fisher_info(self):
            n,K = self.probs.T.shape
            FI = np.zeros((n,K,K))
            d = np.einsum('jii->ij', FI)
            d[:] = 1/self.probs
            return FI

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
            return n/len(Y)    

    return Categorical