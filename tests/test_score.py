# pylint: disable=unnecessary-lambda-assignment
from typing import List, Tuple

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import approx_fprime

from ngboost.distns import (
    Beta,
    BetaBernoulli,
    BetaBinomial,
    BetaBinomialEstN,
    Cauchy,
    Distn,
    Exponential,
    Gamma,
    HalfNormal,
    Laplace,
    LogitNormal,
    LogNormal,
    MultivariateNormal,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    Poisson,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
)
from ngboost.manifold import manifold
from ngboost.scores import CRPScore, LogScore, Score

DistScore = Tuple[Distn, Score]


def sample_metric(manifold_obj, n_mc_samples=1000):
    """
    Copied from LogScore.
    This can be removed if LogScore.metric is accessible
    after inheritance

    Args:
        n_mc_samples: Number of samples to estimate metric with
        manifold_obj (Manifold): A manifold with d_score and sample attributes

    Returns:
        An estimate of the fisher information

    """
    grads = np.stack(
        [manifold_obj.d_score(np.array([Y])) for Y in manifold_obj.sample(n_mc_samples)]
    )
    return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)


def estimate_grad_err(params: np.ndarray, manifold_test):
    """
    Args:
        params: Initial parameters to test around
        manifold_test (Manifold): The manifold to obtain errors for

    Returns:
        grad_err: An gradient error estimate using finite differences
                    and manifold_test.d_score()

    """
    D = manifold_test(params)
    # The y data to evaluate the score/gradient at
    y = D.sample(1)

    # The following returns the gradient for parameters x
    grad = lambda x: manifold_test(x.reshape(-1, 1)).score(y)
    grad_approx = approx_fprime(params.flatten(), grad, 1e-6)
    grad_true = D.d_score(y)
    err = np.linalg.norm(grad_approx - grad_true)
    grad_err = err / D.n_params
    return grad_err


def estimate_metric_err(params: np.ndarray, manifold_test):
    """
    Args:
        params: Initial parameters to test around
        manifold_test (Manifold): The manifold to obtain errors for

    Returns:
        metric_err: An error estimate between a sampling estimate of metric
                    and manifold_test.metric()
    """
    D = manifold_test(params)
    metric_est = sample_metric(D, n_mc_samples=10000)
    metric_true = D.metric()

    # Normalize by the norm of the true metric
    metric_err = np.linalg.norm((metric_est - metric_true)) / np.linalg.norm(
        metric_true
    )
    return metric_err


def estimate_crps_metric(params: np.ndarray, dist, lower=-np.inf, to_y=None):
    """
    Args:
        params: Parameters to compute the metric at
        dist (Distn): The distribution, which must have a cdf method
        lower: Lower limit of the variable that the CRPS integrates over
        to_y: Maps that variable to the outcome, if the two differ

    Returns:
        The CRPS metric 2 * integral of grad F grad F^T (Dawid, 2007), with
        grad F from finite differences of the cdf
    """
    h = 1e-6
    steps = h * np.eye(len(params))[:, :, np.newaxis]

    def integrand(z, i, j):
        # to_y can overflow to inf in the far tail, where the cdf is 1
        with np.errstate(over="ignore"):
            y = z if to_y is None else to_y(z)
        grad = [
            (dist(params + step).cdf(y) - dist(params - step).cdf(y)).item() / (2 * h)
            for step in steps
        ]
        return grad[i] * grad[j]

    metric = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            metric[i, j] = 2 * quad(integrand, lower, np.inf, args=(i, j))[0]
    return metric


def idfn(dist_score: DistScore):
    dist, score = dist_score
    return dist.__name__ + "_" + score.__name__


# test_dists_metric checks the LogScore metric (the Fisher information), and
# test_crps_metric checks the CRPScore metrics
TEST_METRIC: List[DistScore] = [
    (Normal, LogScore),
    (HalfNormal, LogScore),
    (TFixedDfFixedVar, LogScore),
    (Laplace, LogScore),
    (Poisson, LogScore),
    (Gamma, LogScore),
    (Weibull, LogScore),
    (Beta, LogScore),
    (BetaBernoulli, LogScore),
] + [(MultivariateNormal(i), LogScore) for i in range(2, 5)]
# Fill in the dist, score pair to test the gradient
# Tests all in TEST_METRIC by default
TEST_GRAD: List[DistScore] = TEST_METRIC + [
    (Cauchy, LogScore),
    (T, LogScore),
    (TFixedDf, LogScore),
    (LogitNormal, LogScore),
    (BetaBinomial, LogScore),
    (BetaBinomialEstN, LogScore),
    (Normal, CRPScore),
]

# Each case gives a distribution with a CRPScore and the range and change of
# variable for its CRPS integral. The LogNormal CRPS is computed on log(T).
TEST_CRPS_METRIC = [
    (Normal, -np.inf, None),
    (NormalFixedMean, -np.inf, None),
    (NormalFixedVar, -np.inf, None),
    (Laplace, -np.inf, None),
    (Exponential, 0, None),
    (LogNormal, -np.inf, np.exp),
]


@pytest.mark.parametrize(
    "dist,lower,to_y",
    TEST_CRPS_METRIC,
    ids=[case[0].__name__ for case in TEST_CRPS_METRIC],
)
def test_crps_metric(dist, lower, to_y):
    # the scale is away from 1, so a metric off by a factor of scale fails
    params = np.array([[0.3], [0.7]])[-dist.n_params :]
    # survival distributions only implement censored scores
    if hasattr(dist, "censored_scores"):
        score_dist = dist.uncensor(CRPScore)
    else:
        score_dist = dist
    metric = manifold(CRPScore, score_dist)(params).metric()[0]
    expected = estimate_crps_metric(params, dist, lower, to_y)
    np.testing.assert_allclose(metric, expected, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("dist_score_pair", TEST_GRAD, ids=idfn)
def test_dists_grad(dist_score_pair: DistScore):
    # Set seed as this test involves randomness
    # All errors around 1e-5 mark
    np.random.seed(9)
    dist, score = dist_score_pair
    params = np.random.rand(dist.n_params, 1)
    manifold_test = manifold(score, dist)
    grad_err = estimate_grad_err(params, manifold_test)
    assert grad_err < 1e-3
    # TODO: Laplace CRPScore currently fails this test


@pytest.mark.slow
@pytest.mark.parametrize("dist_score_pair", TEST_METRIC, ids=idfn)
@pytest.mark.parametrize("seed", [18])
def test_dists_metric(dist_score_pair: DistScore, seed: int):
    # Set seed as this test involves randomness
    # Note if this test fails on a new distribution
    # There is a chance it is due to randomness
    np.random.seed(seed)
    dist, score = dist_score_pair
    params = np.random.rand(dist.n_params, 1)
    manifold_test = manifold(LogScore, dist)
    FI_err = estimate_metric_err(params, manifold_test)
    assert FI_err < 1e-1
    # TODO: TFixedDF, Cauchy currently fail this test
