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
        self.dCovdL = self.D_cov_D_L()

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def D_cov_D_L(self):
        commutation = np.zeros((self.p**2, self.p**2))
        ind = np.arange(self.p**2).reshape(self.p, self.p).T.flatten()
        commutation[np.arange(self.p**2), ind] = 1.
        dCovdL = (np.identity(self.p**2) + commutation) @\
                 np.kron(self.L, np.identity(self.p))
        dCovdL = dCovdL.reshape(-1,2,2,2,2).swapaxes(-2,-1)
        return dCovdL

		def nll(self, Y):
        diff = Y - self.loc
        M = diff[:,None,:] @ self.cov_inv @ diff[:,:,None]
        half_log_det = np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(-1)
        const = self.p / 2 * np.log(2*np.pi)
        logpdf = - const - half_log_det - 0.5 * M.flatten()
        return -logpdf

		def D_nll(self, Y_):
				pass

	  def fisher_info(self):
				pass

		def fit(Y):	
				pass
