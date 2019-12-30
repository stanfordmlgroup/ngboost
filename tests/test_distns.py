import unittest

from ngboost.distns import Bernoulli, Normal


class TestDistns(unittest.TestCase):
    def test_normal(self):
        assert Normal.n_params == 2

    def test_bernoulli(self):
        assert Bernoulli.n_params == 1
