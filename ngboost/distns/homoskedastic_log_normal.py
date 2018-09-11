from torch.distributions import LogNormal
from torch.distributions import constraints



class HomoskedasticLogNormal(LogNormal):

    arg_constraints = {'loc': constraints.real }

    def __init__(self, loc, validate_args=None):
        super(HomoskedasticLogNormal, self).__init__(loc, 1.0, validate_args)
