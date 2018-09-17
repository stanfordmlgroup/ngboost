import torch
from torch.distributions import Categorical
from torch.distributions import constraints


class CategoricalDistribution(Categorical):

    def __init__(self, *args):
        logits = torch.stack(args, dim=-1)
        super(CategoricalDistribution, self).__init__(logits=logits)


def get_categorical_distn(K):
    CategoricalDistribution.arg_constraints = {
        "logit_%d" % p: constraints.real for p in range(K)
    }
    return CategoricalDistribution
