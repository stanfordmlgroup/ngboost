import scipy as sp
import numpy as np

class MultivariateNormal(object):
		# TODO: make n_params general
		n_params = 5

    def __init__(self, params, temp_scale = 1.0):
        self.N, p = params.shape
        self.p = int(0.5 * (np.sqrt(8 * p - 3))

        self.loc = params[0:p, :].T

        self.L = np.zeros((self.p, self.p, self.N))
        self.L[np.tril_indices(self.p)] = params[self.p:, :]
        self.L = np.transpose(self.L, (2, 0, 1))
        self.cov = self.L @ np.transpose(self.L, (0, 2, 1))
        self.cov_inv = np.linalg.inv(self.cov)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

		def nll(self, Y):
				pass

		def D_nll(self, Y_):
				pass

	  def fisher_info(self):
				pass

		def fit(Y):	
				pass
