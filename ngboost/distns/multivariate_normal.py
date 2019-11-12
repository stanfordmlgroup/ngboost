import scipy as sp
import numpy as np

from scipy.stats import norm as dist

class MultivariateNormal(object):
    # TODO: make n_params general
    n_params = 5

    def __init__(self, params, temp_scale = 1.0):
        self.n_params, self.N = params.shape
        self.p = int(0.5 * (np.sqrt(8 * self.n_params + 9)- 3))

        self.loc = params[:self.p, :].T

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
        # create commutation matrix
        commutation = np.zeros((self.p**2, self.p**2))
        ind = np.arange(self.p**2).reshape(self.p, self.p).T.flatten()
        commutation[np.arange(self.p**2), ind] = 1.

        # compute Jacobian
        dCovdL = (np.identity(self.p**2) + commutation) @\
                 np.kron(self.L, np.identity(self.p))
        dCovdL = dCovdL.reshape(-1,2,2,2,2).swapaxes(-2,-1)
        return dCovdL

    def nll(self, Y):
        try:
            E = Y['Event']
            T = np.log(Y['Time'])
            
            mu0_given_1, mu1_given_0, var0_given_1, var1_given_0 = self.conditional_dist(T)

            cens = (1-E) * (dist.logpdf(T, loc=self.loc[:,1], scale=self.cov[:,1,1]**0.5) + \
                            np.log(1 - dist.cdf(T, loc=mu0_given_1, scale=var0_given_1**0.5)))
            uncens = E * (dist.logpdf(T, loc=self.loc[:,0], scale=self.cov[:,0,0]**0.5) + \
                          np.log(1 - dist.cdf(T, loc=mu1_given_0, scale=var1_given_0**0.5)))
            return -(cens + uncens)
        except:
            diff = Y - self.loc
            M = diff[:,None,:] @ self.cov_inv @ diff[:,:,None]
            half_log_det = np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(-1)
            const = self.p / 2 * np.log(2*np.pi)
            logpdf = - const - half_log_det - 0.5 * M.flatten()
            return -logpdf

    def D_nll(self, Y_):
        try:
            E = Y_['Event']
            T = np.log(Y_['Time'])

            mu0_given_1, mu1_given_0, var0_given_1, var1_given_0 = self.conditional_dist(T)
            mu0 = self.loc[:,0]
            mu1 = self.loc[:,1]
            var0 = self.cov[:,0,0]
            var1 = self.cov[:,1,1]
            cov = self.cov[:,0,1]
    
            # reshape Jacobian
            tril_indices = np.tril_indices(self.p)
            J = self.dCovdL[:,:,:,tril_indices[0],tril_indices[1]]
            J = np.transpose(J, (0,3,1,2)).reshape(self.N, -1, self.p**2)
            J = J.swapaxes(-2,-1)

            # compute grad mu
            D = np.zeros((self.N, J.shape[-1] + self.p))
            pdf0 = dist.pdf(T, loc=mu0_given_1, scale=var0_given_1**0.5)
            cdf0 = dist.cdf(T, loc=mu0_given_1, scale=var0_given_1**0.5)
            pdf1 = dist.pdf(T, loc=mu1_given_0, scale=var1_given_0**0.5)
            cdf1 = dist.cdf(T, loc=mu1_given_0, scale=var1_given_0**0.5)
            cens_mu0 = (1-E) * (pdf0 / (1-cdf0))
            uncens_mu0 = E * ((T - mu0)/var0 - pdf1 / (1-cdf1) * cov * (1/var0))
            cens_mu1 = (1-E) * (pdf1 / (1-cdf1))
            uncens_mu1 = E * ((T - mu1)/var1 - pdf0 / (1-cdf0) * cov * (1/var1))
            D[:,0] = -(cens_mu0 + uncens_mu0)
            D[:,1] = -(cens_mu1 + uncens_mu1)
      
            # compute grad sigma
            import pdb
            pdb.set_trace()

            return D

        except:
            # reshape Jacobian
            tril_indices = np.tril_indices(self.p)
            J = self.dCovdL[:,:,:,tril_indices[0],tril_indices[1]]
            J = np.transpose(J, (0,3,1,2)).reshape(self.N, -1, self.p**2)
            J = J.swapaxes(-2,-1)

            # compute grad mu
            D = np.zeros((self.N, J.shape[-1] + self.p))
            sigma_inv = np.linalg.inv(self.cov)
            diff = self.loc - Y_
            D[:, :self.p] = (sigma_inv @ diff[:,:,None])[...,0]

            # compute grad sigma
            D_sigma = 0.5*(sigma_inv - sigma_inv @ (diff[:,:,None]*diff[:,None,:]) @ sigma_inv)
            D_sigma = D_sigma.reshape(self.N, -1)
            D_L = J.swapaxes(-2,-1) @ D_sigma[:,:,None]
            D[:, self.p:] = D_L[..., 0]

            return D

    def fisher_info(self):
        # reshape Jacobian
        tril_indices = np.tril_indices(self.p)
        J = self.dCovdL[:,:,:,tril_indices[0],tril_indices[1]]

        FI = np.zeros((self.N,self.n_params,self.n_params))

        # compute FI mu
        FI[:,:self.p,:self.p] = self.cov_inv

        # compute FI sigma
        M = np.einsum('nij,njkl->nikl', self.cov_inv, J)
        M = np.einsum('nijx,njky->nikxy', M, M)
        FI[:,self.p:,self.p:] = 0.5 * np.trace(M, axis1=1, axis2=2)

        return FI

    def conditional_dist(self, Y):
        mu0 = self.loc[:,0]
        mu1 = self.loc[:,1]
        var0 = self.cov[:,0,0]
        var1 = self.cov[:,1,1]
        cov = self.cov[:,0,1]

        mu0_given_1 = mu0 + cov * (1/var1) * (Y - mu1)
        mu1_given_0 = mu1 + cov * (1/var0) * (Y - mu0)
        var0_given_1 = var0 - cov * (1/var1) * cov
        var1_given_0 = var1 - cov * (1/var0) * cov

        return mu0_given_1, mu1_given_0, var0_given_1, var1_given_0

    def fit(Y):	
        try:
            E = Y['Event']
            T = np.log(Y['Time'])
            # place holder
            m = np.array([8., 8.])
            sigma = np.array([[1., .5],
                              [.5, 1.]])
            L = sp.linalg.cholesky(sigma, lower=True)
            return np.concatenate([m, L[np.tril_indices(2)]])
        except:
            N, p = Y.shape
            m = Y.mean(axis=0)
            diff = Y - m
            sigma = 1 / N * (diff[:,:,None] @ diff[:,None,:]).sum(0)
            L = sp.linalg.cholesky(sigma, lower=True)
            return np.concatenate([m, L[np.tril_indices(p)]])
