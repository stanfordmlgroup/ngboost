from torch.distributions import Normal
from torch.distributions import constraints



class HomoskedasticNormal(Normal):

    arg_constraints = {'loc': constraints.real }

    def __init__(self, loc, validate_args=None):
        super(HomoskedasticNormal, self).__init__(loc, 1.0, validate_args)
