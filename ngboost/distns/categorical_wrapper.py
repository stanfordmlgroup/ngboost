import torch
from torch.distributions import Categorical
from torch.distributions import constraints


class CategoricalWrapper(Categorical):

    def __init__(self, *args):
        logits = torch.stack(args, dim=-1)
        super(CategoricalWrapper, self).__init__(logits=logits)


def get_categorical_distn(n_categories):
    CategoricalWrapper.arg_constraints = {
        "logit_%d" % p: constraints.real for p in range(n_categories)}
    return CategoricalWrapper
