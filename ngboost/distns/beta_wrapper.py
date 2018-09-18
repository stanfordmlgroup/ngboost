import torch
from torch.distributions import Beta, Bernoulli


class BetaWrapper(Beta):
    """
    Wrapper around a latent Beta distribution to model binary outcomes,
    with optimization using using Monte Carlo gradients.
    """

    n_mc_samples = 50

    def log_prob(self, value):
        samples = self.rsample(
            (value.shape[0], BetaWrapper.n_mc_samples))
        samples = samples.mean(dim=1)
        return Bernoulli(probs=samples).log_prob(value)


def get_beta_distn(n_mc_samples):
    BetaWrapper.n_mc_samples = n_mc_samples
    return BetaWrapper
