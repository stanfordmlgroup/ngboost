"""The NGBoost base distribution"""
from warnings import warn

from jax import grad, vmap
import jax.numpy as np
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
    @classmethod
    def has(cls, *attributes):
        return all(hasattr(cls, attribute) for attribute in attributes)

    def __init__(self, params):
        self._params = params

    def __getitem__(self, key):
        return self.__class__(self._params[:, key])

    def __len__(self):
        return self._params.shape[1]

    @classmethod
    def parametrize_internally(cls, fun):
        return lambda _params, Y: fun(Y, **cls.params_to_user(_params))

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
    def find_implementation(cls, Score):
        """
        Finds the distribution-appropriate implementation of Score
        """

        try:
            return {*cls.__subclasses__()}.intersection({*Score.__subclasses__()}).pop()
        except KeyError as err:
            if Score.__bases__[-1] is ScoreRoot and Score.from_scratch:
                # build a manifold inheriting from dist and score on-the-fly
                return type(f"{cls.__name__}{Score.__name__}Manifold", (cls, Score), {})
            else:
                raise ValueError(
                    f"The scoring rule {Score.__name__} is not "
                    f"implemented for the {cls.__name__} distribution and cannot be computationally derived"
                )


class RegressionDistn(Distn):
    pass


class ClassificationDistn(Distn):
    def predict(self):  # returns class assignments
        return np.argmax(self.class_probs(), 1)
