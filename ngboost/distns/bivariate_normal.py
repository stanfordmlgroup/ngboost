import scipy as sp
import numpy as np
from scipy.stats import norm as dist
from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore, CRPScore

eps = 1e-6


class BivariateNormalCensoredScore(LogScore):

    def score(self, Y):
        E, T = Y['Event'], Y['Time']
        nll = - self.marginal_dist(Y).logpdf(T) - self.conditional_dist(Y).logsf(T) + self.censoring_prob(Y)
        return nll

    def d_score(self, Y):
        D = - self.D_conditional_logsf(Y) - self.D_marginal_logpdf(Y) + self.D_censoring_prob(Y)
        return D

    def metric(self):
        J = np.zeros((2,2,3,self.N))
        J[0,0,0] = 2*self.var0
        J[0,1,0] = J[1,0,0] = self.cor
        J[1,1,1] = 2*self.var1
        J[0,1,1] = J[1,0,1] = self.cor
        J[0,1,2] = J[1,0,2] = self.scale0*self.scale1/(1.+self.s**2)

        FI = np.zeros((self.N, self.n_params, self.n_params))

        S = np.array([[self.var0, self.cov], [self.cov, self.var1]])
        S = np.transpose(S, (2,0,1))
        invS = np.linalg.inv(S)

        FI[:,:2,:2] = invS

        J = np.transpose(J, (3,0,1,2))
        M = np.einsum('nij,njkl->nikl', invS, J)
        M = np.einsum('nijx,njky->nikxy', M, M)
        FI[:,2:,2:] = 0.5*np.trace(M, axis1=1, axis2=2)

        return FI

    def conditional_dist(self, Y):
        E, T = Y['Event'], Y['Time']
        cond_mu = E * (self.mu0 + self.cov / self.var1 * (T - self.mu1)) +\
                  (1-E) * (self.mu1 + self.cov / self.var0 * (T - self.mu0))
        cond_var = E * (self.var0 - self.cov**2 / self.var1) +\
                   (1-E) * (self.var1 - self.cov**2 / self.var0) + 1e-4
        return dist(loc=cond_mu, scale=cond_var**(1/2))

    def D_conditional_logsf(self, Y):
        E, T = Y['Event'], Y['Time']
        D = np.zeros((self.N, self.n_params))
        eps = 1e-8

        cond_dist = self.conditional_dist(Y)
        Z = (T-cond_dist.mean()) / cond_dist.std()
        pdf = dist().pdf(Z)
        sf = dist().sf(Z)
        dZ = pdf / (sf+eps)

        D[:,0] = dZ / cond_dist.std() * (-self.cov/self.var0)**(1-E)
        D[:,1] = dZ / cond_dist.std() * (-self.cov/self.var1)**E

        dMu_dCov = E*(T-self.mu1)/self.var1 + (1-E)*(T-self.mu0)/self.var0
        dStd_dCov = -2*self.cov / self.var1**E / self.var0**(1-E) / (2*cond_dist.std())
        dCov = (-cond_dist.std()*dMu_dCov - (T-cond_dist.mean())*dStd_dCov)/cond_dist.var()

        dMu_dVar0 = -self.cov * (T-self.mu0) / self.var0**2
        dStd_dVar0 = self.cov**2 / self.var0**2 / (2*cond_dist.std())
        dVar0 = E * (-Z / cond_dist.var() / 2) +\
                (1-E) * (-cond_dist.std()*dMu_dVar0 - (T-cond_dist.mean())*dStd_dVar0)/cond_dist.var()

        dMu_dVar1 = -self.cov * (T-self.mu1) / self.var1**2
        dStd_dVar1 = self.cov**2 / self.var1**2 / (2*cond_dist.std())
        dVar1 = (1-E) * (-Z / cond_dist.var() / 2) +\
                E * (-cond_dist.std()*dMu_dVar1 - (T-cond_dist.mean())*dStd_dVar1)/cond_dist.var()

        D[:,2] = dZ * (-dVar0 * 2 * self.var0 - dCov * self.cov)
        D[:,3] = dZ * (-dVar1 * 2 * self.var1 - dCov * self.cov)
        D[:,4] = dZ * -dCov * self.scale0 * self.scale1 * (1./(1.+self.s**2))

        return D

    def marginal_dist(self, Y):
        E, T = Y['Event'], Y['Time']
        marg_mu = E * self.mu1 + (1-E) * self.mu0
        marg_scale = E * self.scale1 + (1-E) * self.scale0
        return dist(loc=marg_mu, scale=marg_scale)

    def D_marginal_logpdf(self, Y):
        E, T = Y['Event'], Y['Time']
        D = np.zeros((self.N, self.n_params))
        D[:,0] = (1-E) * (T - self.mu0) / self.var0
        D[:,1] = E * (T - self.mu1) / self.var1
        D[:,2] = (1-E)*(-1 + ((T - self.mu0)**2 / self.var0))
        D[:,3] = E * (-1 + ((T - self.mu1)**2 / self.var1))
        return D

    def censoring_prob(self, Y):
        E, T = Y['Event'], Y['Time']
        mu = (self.mu0 - self.mu1) * (-1)**E
        var = self.var0 + self.var1 - 2*self.cov
        Z = dist(loc=mu, scale=var**(1/2))
        return Z.logcdf(0.)

    def D_censoring_prob(self, Y):
        E, T = Y['Event'], Y['Time']
        D = np.zeros((self.N, self.n_params))

        mu = (self.mu0 - self.mu1) * (-1)**E
        var = self.var0 + self.var1 - 2*self.cov
        pdf = dist().pdf(-mu/var**(1/2))
        cdf = dist().cdf(-mu/var**(1/2))

        D[:,0] = -pdf / cdf /  var**(1/2) * (-1)**E
        D[:,1] = -pdf / cdf / var**(1/2) * (-1)**(1-E)
        D[:,2] = pdf / cdf * mu / (2*var**(3/2)) * (2*self.var0 - 2*self.cov)
        D[:,3] = pdf / cdf * mu / (2*var**(3/2)) * (2*self.var1 - 2*self.cov)
        D[:,4] = -2*pdf / cdf * mu / (2*var**(3/2)) * self.scale0 * self.scale1 * (1./(1. + self.s**2))

        return D


class BivariateNormal(RegressionDistn):

    n_params = 5
    censored_scores = [BivariateNormalCensoredScore]
    # Event = 1 means event
    # Event = 0 means censored
    # mu1 = Event
    # mu0 = Censored

    def __init__(self, params):
        self._params = params
        self.n_params, self.N = params.shape
        self.mu0, self.mu1 = params[0], params[1]
        self.scale0, self.scale1 = np.exp(params[2]), np.exp(params[3])
        self.var0, self.var1 = self.scale0**2, self.scale1**2
        self.s = params[4]
        self.cor = np.tanh(self.s)
        self.cov = self.cor * self.scale0 * self.scale1

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def mean(self):
        return self.mu0

    def var(self):
        return self.var0

    def ppf(self, x):
        return sp.stats.norm(self.mu0, self.var0).ppf(x)

    def cdf(self, x):
        return sp.stats.norm(self.mu0, self.var0).cdf(x)

    @classmethod
    def fit(self, Y):
#        m, s = sp.stats.norm.fit(np.log(Y))
        m, s = sp.stats.norm.fit(Y)
        return np.array([m, np.log(s), m, np.log(s), 0])

