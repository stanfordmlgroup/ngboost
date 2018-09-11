
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform


class AffineDistribution(TransformedDistribution):
    """
    Wrapper around an affine-transformed distribution.

    Needed in order to support mean and variance functions.
    Differential entropy is already taken care of.
    """
    def __init__(self, base_distribution, transform):
        self.base_distribution = base_distribution
        self.transform = transform
        super(AffineDistribution, self).__init__(base_distribution, [transform])

    @property
    def mean(self):
        return self.transform.loc + self.transform.scale * \
                                    self.base_distribution.mean

    @property
    def variance(self):
        return self.transform.scale ** 2 + self.base_distribution.variance
