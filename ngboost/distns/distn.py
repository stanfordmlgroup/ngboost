"""The NGBoost base distribution"""
from warnings import warn

from jax import grad, vmap
import jax.numpy as np
from toolz.functoolz import compose
from dataclasses import dataclass

from inspect import signature

from ngboost.scores import Score as ScoreRoot


@dataclass
class IntervalParameter:
    min: np.float32 = -np.inf
    max: np.float32 = np.inf

    def to_internal(self, param):
        scaled = (param - self.min) / (self.max - self.min)
        positive = scaled / (1 - scaled)
        return np.log(positive)

    def to_user(self, _param):
        positive = np.exp(_param)
        scaled = positive / (1 + positive)
        return scaled * (self.max - self.min) + self.min


@dataclass
class UpperBoundParameter:
    max: np.float32 = np.inf

    def to_internal(self, param):
        positive = self.max - param
        return np.log(positive)

    def to_user(self, _param):
        positive = np.exp(_param)
        return self.max - positive


@dataclass
class LowerBoundParameter:
    min: np.float32 = np.inf

    def to_internal(self, param):
        positive = param - self.min
        return np.log(positive)

    def to_user(self, _param):
        positive = np.exp(_param)
        return self.min + positive


@dataclass
class RealParameter:
    def to_internal(self, param):
        return param

    def to_user(self, _param):
        return _param


def Parameter(min=None, max=None):
    if min is None and max is None:
        return RealParameter()
    elif min is None:
        return UpperBoundParameter(max=max)
    elif max is None:
        return LowerBoundParameter(min=min)
    else:
        return IntervalParameter(min=min, max=max)


class Distn:
    # functions that are like _fn operate on the internal array parametrization

    def __init__(self, params):
        self._params = params

    def __getitem__(self, key):
        return self.__class__(self._params[:, key])

    def __len__(self):
        return self._params.shape[1]

    @classmethod
    def parametrize_internally(cls, fun):
        return lambda Y, _params: fun(Y, **cls.params_to_user(_params))

    @classmethod
    def params_to_user(cls, _params):
        return {
            param_name: parametrization.to_user(_param)
            for (param_name, parametrization), _param in zip(
                cls.parametrization.items(), _params.T
            )
        }

    @classmethod
    def params_to_internal(cls, *param_list, **param_dict):
        if len(param_list) > 0 and len(param_dict) > 0:
            raise ValueError(
                "Params must either be passed as array or dictionary, not mixed"
            )

        if len(param_list) > 0:
            param_dict = dict(zip(cls.parametrization.keys(), param_list))

        return np.array(
            [
                cls.parametrization[param_name].to_internal(param)
                for param_name, param in param_dict.items()
            ]
        ).T

    @classmethod
    def n_params(cls):
        return len(cls.parametrization)

    @property
    def params(self):
        return self.params_to_user(self._params)

    @classmethod
    def implementation(cls, Score, scores=None):
        """
        Finds the distribution-appropriate implementation of Score
        (using the provided scores if cls.scores is empty)
        """
        if scores is None:
            scores = cls.scores
        if Score.__bases__[-1] is ScoreRoot and Score in scores:
            return Score
        try:
            return {S.__bases__[-1]: S for S in scores}[Score]
        except KeyError as err:
            raise ValueError(
                f"The scoring rule {Score.__name__} is not "
                f"implemented for the {cls.__bases__[-1].__name__} distribution."
            ) from err


class RegressionDistn(Distn):
    @classmethod
    def build(cls):
        class BuiltDist(cls):

            if not hasattr(cls, "cdf"):
                raise ValueError(
                    f"The distribution {cls.__name__} has no CDF defined."
                    f" clsributions for regression must define at least a CDF."
                )

            if not hasattr(cls, "_cdf"):
                _cdf = cls.parametrize_internally(cls.cdf)

            if not hasattr(cls, "_likelihood"):
                _likelihood = grad(_cdf)

            if not hasattr(cls, "_nll"):
                if hasattr(cls, "nll"):
                    _nll = cls.parametrize_internally(cls.nll)
                else:
                    _nll = compose(lambda x: -x, np.log, _likelihood)

        return BuiltDist


class ClassificationDistn(Distn):
    def predict(self):  # returns class assignments
        return np.argmax(self.class_probs(), 1)
