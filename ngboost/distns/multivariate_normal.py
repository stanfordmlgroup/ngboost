"""The NGBoost multivariate Normal distribution and scores"""
# pylint: disable=unused-argument,bare-except,too-many-locals
import numpy as np
import scipy as sp
from scipy.stats import norm as dist

eps = 1e-6


class MultivariateNormal:  # pylint: disable=too-many-instance-attributes
    # make n_params general
    n_params = 5

    def __init__(self, params, temp_scale=1.0):
        self.n_params, self.N = params.shape
        self.p = int(0.5 * (np.sqrt(8 * self.n_params + 9) - 3))

        self.loc = params[: self.p, :].T

        self.L = np.zeros((self.p, self.p, self.N))
        self.L[np.tril_indices(self.p)] = params[self.p :, :]
        self.L = np.transpose(self.L, (2, 0, 1))
        self.cov = self.L @ np.transpose(self.L, (0, 2, 1))
        self.cov_inv = np.linalg.inv(self.cov)
        self.dCovdL = self.D_cov_D_L()

    def mean(self):
        return np.exp(self.loc[:, 0])

    def D_cov_D_L(self):
        # create commutation matrix
        commutation = np.zeros((self.p ** 2, self.p ** 2))
        ind = np.arange(self.p ** 2).reshape(self.p, self.p).T.flatten()
        commutation[np.arange(self.p ** 2), ind] = 1.0

        # compute Jacobian
        dCovdL = (np.identity(self.p ** 2) + commutation) @ np.kron(
            self.L, np.identity(self.p)
        )
        dCovdL = dCovdL.reshape(-1, 2, 2, 2, 2).swapaxes(-2, -1)
        return dCovdL

    def nll(self, Y):
        try:
            E = Y["Event"]
            T = np.log(Y["Time"] + eps)

            (
                mu0_given_1,
                mu1_given_0,
                var0_given_1,
                var1_given_0,
            ) = self.conditional_dist(T)

            cens = (1 - E) * (
                dist.logpdf(T, loc=self.loc[:, 1], scale=self.cov[:, 1, 1] ** 0.5)
                + np.log(
                    eps + 1 - dist.cdf(T, loc=mu0_given_1, scale=var0_given_1 ** 0.5)
                )
            )
            uncens = E * (
                dist.logpdf(T, loc=self.loc[:, 0], scale=self.cov[:, 0, 0] ** 0.5)
                + np.log(
                    eps + 1 - dist.cdf(T, loc=mu1_given_0, scale=var1_given_0 ** 0.5)
                )
            )
            return -(cens + uncens)
        except:  # NOQA
            diff = Y - self.loc
            M = diff[:, None, :] @ self.cov_inv @ diff[:, :, None]
            half_log_det = np.log(eps + np.diagonal(self.L, axis1=1, axis2=2)).sum(-1)
            const = self.p / 2 * np.log(eps + 2 * np.pi)
            logpdf = -const - half_log_det - 0.5 * M.flatten()
            return -logpdf

    # pylint: disable=too-many-statements
    def D_nll(self, Y_):
        try:
            E = Y_["Event"]
            T = np.log(Y_["Time"] + eps)

            (
                mu0_given_1,
                mu1_given_0,
                var0_given_1,
                var1_given_0,
            ) = self.conditional_dist(T)
            mu0 = self.loc[:, 0]
            mu1 = self.loc[:, 1]
            var0 = self.cov[:, 0, 0]
            var1 = self.cov[:, 1, 1]
            cov = self.cov[:, 0, 1]

            # reshape Jacobian
            tril_indices = np.tril_indices(self.p)
            J = self.dCovdL[:, :, :, tril_indices[0], tril_indices[1]]
            J = np.transpose(J, (0, 3, 1, 2)).reshape(self.N, -1, self.p ** 2)
            J = J.swapaxes(-2, -1)

            # compute grad mu
            D = np.zeros((self.N, self.n_params))
            diff0 = T - mu0
            diff1 = T - mu1
            Z0 = diff0 / (var0 ** 0.5 + eps)
            Z1 = diff1 / (var1 ** 0.5 + eps)
            Z0_1 = (T - mu0_given_1) / (var0_given_1 ** 0.5 + eps)
            Z1_0 = (T - mu1_given_0) / (var1_given_0 ** 0.5 + eps)
            pdf0 = dist.pdf(T, loc=mu0_given_1, scale=var0_given_1 ** 0.5)
            cdf0 = dist.cdf(T, loc=mu0_given_1, scale=var0_given_1 ** 0.5)
            pdf1 = dist.pdf(T, loc=mu1_given_0, scale=var1_given_0 ** 0.5)
            cdf1 = dist.cdf(T, loc=mu1_given_0, scale=var1_given_0 ** 0.5)
            cens_mu0 = (1 - E) * (pdf0 / (eps + 1 - cdf0))
            uncens_mu0 = E * (
                diff0 / (eps + var0) - pdf1 / (eps + 1 - cdf1) * (cov / (eps + var0))
            )
            uncens_mu1 = E * (pdf1 / (eps + 1 - cdf1))
            cens_mu1 = (1 - E) * (
                diff1 / (eps + var1) - pdf0 / (eps + 1 - cdf0) * (cov / (eps + var1))
            )
            D[:, 0] = -(cens_mu0 + uncens_mu0)
            D[:, 1] = -(cens_mu1 + uncens_mu1)

            # compute grad sigma
            D_sigma = np.zeros((self.N, self.p ** 2))

            cens_var0 = (1 - E) * (
                pdf0 / (eps + 1 - cdf0) * Z0_1 / (eps + 2 * var0_given_1 ** 0.5)
            )
            uncens_var0 = E * (
                0.5 * ((Z0 ** 2 - 1) / (eps + var0))
                + pdf1
                / (eps + 1 - cdf1)
                * (
                    -Z0 * (cov / (eps + var0 ** 1.5))
                    + 0.5
                    * Z1_0
                    / (eps + var1_given_0 ** 0.5)
                    * (cov / (eps + var0)) ** 2
                )
            )
            D_sigma[:, 0] = -(cens_var0 + uncens_var0)

            uncens_var1 = E * (
                pdf1 / (eps + 1 - cdf1) * Z1_0 / (eps + 2 * var1_given_0 ** 0.5)
            )
            cens_var1 = (1 - E) * (
                0.5 * ((Z1 ** 2 - 1) / (eps + var1))
                + pdf0
                / (eps + 1 - cdf0)
                * (
                    -Z1 * (cov / (eps + var1 ** 1.5))
                    + 0.5
                    * Z0_1
                    / (eps + var0_given_1 ** 0.5)
                    * (cov / (eps + var1)) ** 2
                )
            )
            D_sigma[:, 3] = -(cens_var1 + uncens_var1)

            uncens_cov = E * (
                pdf1
                / (eps + 1 - cdf1)
                * (
                    Z0 / (eps + var0 ** 0.5)
                    - Z1_0 / (eps + var1_given_0 ** 0.5) * (cov / (eps + var0))
                )
            )
            cens_cov = (1 - E) * (
                pdf0
                / (eps + 1 - cdf0)
                * (
                    Z1 / (eps + var1 ** 0.5)
                    - Z0_1 / (eps + var0_given_1 ** 0.5) * (cov / (eps + var1))
                )
            )
            D_sigma[:, 1] = -(cens_cov + uncens_cov) * 0.5
            D_sigma[:, 2] = -(cens_cov + uncens_cov) * 0.5

            D_L = J.swapaxes(-2, -1) @ D_sigma[:, :, None]
            D[:, self.p :] = D_L[..., 0]

            return D

        except:  # NOQA
            # reshape Jacobian
            tril_indices = np.tril_indices(self.p)
            J = self.dCovdL[:, :, :, tril_indices[0], tril_indices[1]]
            J = np.transpose(J, (0, 3, 1, 2)).reshape(self.N, -1, self.p ** 2)
            J = J.swapaxes(-2, -1)

            # compute grad mu
            D = np.zeros((self.N, J.shape[-1] + self.p))
            sigma_inv = np.linalg.inv(self.cov)
            diff = self.loc - Y_
            D[:, : self.p] = (sigma_inv @ diff[:, :, None])[..., 0]

            # compute grad sigma
            D_sigma = 0.5 * (
                sigma_inv
                - sigma_inv @ (diff[:, :, None] * diff[:, None, :]) @ sigma_inv
            )
            D_sigma = D_sigma.reshape(self.N, -1)
            D_L = J.swapaxes(-2, -1) @ D_sigma[:, :, None]
            D[:, self.p :] = D_L[..., 0]

            return D

    def fisher_info(self):
        # reshape Jacobian
        tril_indices = np.tril_indices(self.p)
        J = self.dCovdL[:, :, :, tril_indices[0], tril_indices[1]]

        FI = np.zeros((self.N, self.n_params, self.n_params))

        # compute FI mu
        FI[:, : self.p, : self.p] = self.cov_inv

        # compute FI sigma
        M = np.einsum("nij,njkl->nikl", self.cov_inv, J)
        M = np.einsum("nijx,njky->nikxy", M, M)
        FI[:, self.p :, self.p :] = 0.5 * np.trace(M, axis1=1, axis2=2)

        return FI

    def conditional_dist(self, Y):
        mu0 = self.loc[:, 0]
        mu1 = self.loc[:, 1]
        var0 = self.cov[:, 0, 0]
        var1 = self.cov[:, 1, 1]
        cov = self.cov[:, 0, 1]

        mu0_given_1 = mu0 + cov * (1 / (eps + var1)) * (Y - mu1)
        mu1_given_0 = mu1 + cov * (1 / (eps + var0)) * (Y - mu0)
        var0_given_1 = var0 - cov * (1 / (eps + var1)) * cov
        var1_given_0 = var1 - cov * (1 / (eps + var0)) * cov

        return mu0_given_1, mu1_given_0, var0_given_1, var1_given_0

    def fit(Y):
        try:
            # place holder
            m = np.array([8.0, 8.0])
            sigma = np.array([[2.0, 1.0], [1.0, 2.0]])
            L = sp.linalg.cholesky(sigma, lower=True)
            return np.concatenate([m, L[np.tril_indices(2)]])
        except:  # NOQA
            N, p = Y.shape
            m = Y.mean(axis=0)  # pylint: disable=unexpected-keyword-arg
            diff = Y - m
            sigma = 1 / N * (diff[:, :, None] @ diff[:, None, :]).sum(0)
            L = sp.linalg.cholesky(sigma, lower=True)
            return np.concatenate([m, L[np.tril_indices(p)]])
