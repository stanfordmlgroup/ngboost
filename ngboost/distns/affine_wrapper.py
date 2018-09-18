from torch.distributions.transformed_distribution import TransformedDistribution


class AffineWrapper(TransformedDistribution):
    """
    Wrapper around an affine-transformed distribution.
    Needed in order to support mean and variance functions.
    """
    def __init__(self, base_distribution, transform):
        self.base_distribution = base_distribution
        self.transform = transform
        super(AffineWrapper, self).__init__(base_distribution, [transform])

    @property
    def mean(self):
        return self.transform.loc + self.transform.scale * \
                                    self.base_distribution.mean

    @property
    def variance(self):
        return self.transform.scale ** 2 + self.base_distribution.variance
