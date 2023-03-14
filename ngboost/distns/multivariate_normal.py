"""
parameterization and gradient equations inspiration
taken from Using Neural Networks to Model Conditional Multivariate Densities
Peter Martin Williams 1996
"""

import warnings

import numpy as np
import scipy as sp

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


def get_tril_idxs(p):
    tril_indices = np.tril_indices(p)
    mask_diag = tril_indices[0] == tril_indices[1]

    off_diags = np.where(np.invert(mask_diag))[0]
    diags = np.where(mask_diag)[0]

    return tril_indices, diags, off_diags


class MVNLogScore(LogScore):
    def score(self, Y):
        return -self.logpdf(Y)

    def d_score(self, Y):
        """
        Formulas for this part taken from the paper mentioned at the start

        Args:
            Y: The response data

        Returns:
            self.N, self.n_params shaped array containing the gradient.

        """
        diff, eta = self.summaries(Y)

        gradient = np.zeros((self.N, self.n_params))

        # Gradient of the mean
        # N x 1 x p matrix by N x p x p
        grad_mu = np.expand_dims(eta, 1) @ self.L.transpose(0, 2, 1)
        grad_mu = grad_mu.squeeze(axis=1)
        gradient[:, : self.p] = grad_mu

        tril_indices, diags, off_diags = get_tril_idxs(self.p)
        # Gradient of the diagonal
        diagonal_elements = np.diagonal(self.L, axis1=1, axis2=2)
        grad_diag = (
            diff * np.expand_dims(eta, 2) * np.expand_dims(diagonal_elements, 2) - 1
        )
        grad_diag = grad_diag.squeeze(axis=2)
        grad_reference = gradient[:, self.p :]
        grad_reference[:, diags] = grad_diag
        # Off diagonal elements of L gradient:
        for par_idx in off_diags:
            i = tril_indices[0][par_idx]
            j = tril_indices[1][par_idx]
            grad_reference[:, par_idx] = eta[:, j] * diff[:, i, 0]

        return gradient

    def metric(self):

        """
        Formulas for this part are not in the the paper mentioned.
        Obtained by taking the expectation of the Hessian.


        Returns:
            self.N, self.n_params, self.n_params shaped array containing the fisher information for
             the ith observation in the last two indices.

        """
        FisherInfo = np.stack([np.identity(self.n_params)] * self.N)
        # FI of the location
        FisherInfo[:, : self.p, : self.p] = self.L @ self.L.transpose(0, 2, 1)

        # Get VarComp as a reference to the diagonals.
        VarComp = FisherInfo[:, self.p :, self.p :]

        # E[diff_i eta_j] is the following
        cov_sum = self.L.transpose(0, 2, 1) @ self.cov

        tril_indices, diags, off_diags = get_tril_idxs(self.p)

        # Following loop is
        # E[d^2l / dlog(a_ii) dlog(a_kk)]
        # and
        # E[d^2l / dlog(a_ii) dlog(a_kq)]
        # where a_ik = exp(L_ki) if i=k and a_ik=L_ki otherwise.
        for diag_idx in diags:
            i = tril_indices[0][diag_idx]
            value = (
                self.L[:, i, i] ** 2 * self.cov[:, i, i]
                + cov_sum[:, i, i] * self.L[:, i, i]
            )
            VarComp[:, diag_idx, diag_idx] = value
            VarComp[:, diag_idx, diag_idx] = value
            for par_idx in off_diags:
                q = tril_indices[0][par_idx]
                k = tril_indices[1][par_idx]
                if i == k:
                    value = self.cov[:, q, i] * self.L[:, i, i]
                    VarComp[:, diag_idx, par_idx] = value
                    VarComp[:, par_idx, diag_idx] = value

        # Off diagonals  w.r.t. off diagonals
        for par_idx in off_diags:
            j = tril_indices[0][par_idx]
            i = tril_indices[1][par_idx]
            for par_idx2 in off_diags:
                k = tril_indices[0][par_idx2]
                q = tril_indices[1][par_idx2]
                if i == q:
                    value = self.cov[:, k, j]
                    VarComp[:, par_idx, par_idx2] = value
                    VarComp[:, par_idx2, par_idx] = value
        return FisherInfo


def get_chol_factor(lower_tri_vals):
    """

    Args:
        lower_tri_vals: numpy array, shaped as the number of lower triangular
                        elements, number of observations.
                        The values ordered according to np.tril_indices(p)
                        where p is the dimension of the multivariate normal distn

    Returns:
        Nxpxp numpy array, with the lower triangle filled in. The diagonal is exponentiated.

    """
    lower_size, N = lower_tri_vals.shape

    # solve p(p+3)/2 = lower_size to get the
    # number of dimensions.

    p = (-1 + (1 + 8 * lower_size) ** 0.5) / 2
    p = int(p)

    if not isinstance(lower_tri_vals, np.ndarray):
        lower_tri_vals = np.array(lower_tri_vals)

    L = np.zeros((N, p, p))
    for par_ind, (k, l) in enumerate(zip(*np.tril_indices(p))):
        if k == l:
            # Add a small number to avoid singular matrices.
            L[:, k, l] = np.exp(lower_tri_vals[par_ind, :]) + 1e-6
        else:
            L[:, k, l] = lower_tri_vals[par_ind, :]
    return L


def MultivariateNormal(k):
    """
    #  Factory function that generates classes for
    #  k-dimensional multivariate normal distributions for NGBoost

    # This distribution has LogScore implemented for it.

    # Currently only for a regression implementation.
    """
    if k == 1:
        warnings.warn(
            "Using Multivariate normal with k=1. Using ngboost.distn.Normal instead is advised."
        )

    # pylint: disable=too-many-instance-attributes
    class MVN(RegressionDistn):
        """
        Implementation of the Multivariate normal distribution for regression.
        Using the parameterization Sigma^{-1} = LL^T and modelling L
        diag(L) is exponentiated to ensure parameters are unconstrained.

        Scipy's multivariate normal benchmarks were relatively
        slow for pdf calculations so the implementation is from scratch.

        As Scipy has considerably more features call self.scipy_distribution()
        to return a list of distributions.
        """

        n_params = int(k * (k + 3) / 2)
        scores = [MVNLogScore]
        multi_output = True

        # pylint: disable=super-init-not-called
        def __init__(self, params):
            super().__init__(params)
            self.N = params.shape[1]

            # Number of MVN dimensions, =k originally
            self.p = (-3 + (9 + 8 * self.n_params) ** 0.5) / 2
            self.p = int(self.p)

            # Extract parameters from params list
            # Param array is assumed to of shape n_params,N
            # First p rows are the mean
            # rest are the lower triangle of L.
            # Where Sigma_inverse = L@L.transpose()
            # Diagonals modelled on the log scale.
            self.loc = np.transpose(np.array(params[: self.p, :]))
            self.tril_L = np.array(params[self.p :, :])

            # Returns 3d array, shape (p, p, N)
            self.L = get_chol_factor(self.tril_L)

            # The remainder is just for utility.
            self.cov_inv = self.L @ self.L.transpose(0, 2, 1)

            # _cov_mat and _Lcov are place holders, relatively expensive to compute
            # The inverse of self.cov_inv. Access through self.cov
            self._cov_mat = None
            # cholesky factor of _cov_mat, useful for random number generation
            self._Lcov = None
            # Saving the pdf constant and means in an accessible format.
            self.pdf_constant = -self.p / 2 * np.log(2 * np.pi)

        def summaries(self, Y):
            """
            Parameters:
                Y: The data being fit to

            Returns:
                diff: N x p x1 the residual between the mean and the data
                eta: N x p which is L@diff
            """
            diff = np.expand_dims(self.loc - Y, 2)

            # N x 2 x 2 @ N x p x 1
            # -> N x p x 1 we remove the last index
            eta = np.squeeze(np.matmul(self.L.transpose(0, 2, 1), diff), axis=2)
            return diff, eta

        def logpdf(self, Y):
            _, eta = self.summaries(Y)
            # the exponentiated part of the pdf:
            p1 = -0.5 * np.sum(np.square(eta), axis=1)
            p1 = p1.squeeze()
            # this is the sqrt(determinant(Sigma)) component of the pdf
            p2 = np.sum(np.log(np.diagonal(self.L, axis1=1, axis2=2)), axis=1)

            ret = p1 + p2 + self.pdf_constant
            return ret

        def fit(Y):
            N, p = Y.shape
            m = Y.mean(axis=0)  # pylint: disable=unexpected-keyword-arg
            diff = Y - m
            sigma = 1 / N * (diff[:, :, None] @ diff[:, None, :]).sum(0)
            L = sp.linalg.cholesky(np.linalg.inv(sigma), lower=True)
            diag_idx = np.diag_indices(p)
            L[diag_idx] = np.log(L[diag_idx])
            return np.concatenate([m, L[np.tril_indices(p)]])

        def rv(self):
            # This is only useful for generating rvs so only compute it in rv.
            if self._Lcov is None:
                self._Lcov = np.linalg.cholesky(np.linalg.inv(self.cov_inv))

            u = np.random.normal(loc=0, scale=1, size=(self.N, self.p, 1))
            sample = np.expand_dims(self.loc, 2) + self._Lcov @ u
            return np.squeeze(sample)

        def rvs(self, n):
            return [self.rv() for _ in range(n)]

        def sample(self, n):
            return self.rvs(n)

        @property
        def cov(self):
            # Covariance matrix is for computing the fisher information
            # If it is singular it is set an extremely large value.
            # This will probably not affect anything.
            if self._cov_mat is None:
                self._cov_mat = np.linalg.inv(self.cov_inv)
            return self._cov_mat

        @property
        def params(self):
            return {"loc": self.loc, "scale": self.cov}

        def scipy_distribution(self):
            """
            Returns:
                List of scipy.stats.multivariate_normal distributions.
            """
            cov_mat = self.cov
            return [
                sp.stats.multivariate_normal(mean=self.loc[i, :], cov=cov_mat[i, :, :])
                for i in range(self.N)
            ]

        def mean(self):
            return self.loc

    return MVN
