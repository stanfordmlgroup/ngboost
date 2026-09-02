"""The NGBoost Library"""

try:
    from importlib.metadata import version
except ImportError:
    # before python 3.8
    from importlib_metadata import version

from .api import NGBClassifier, NGBRegressor, NGBSurvival
from .helpers import (
    load_ngboost_model,
    load_ngboost_model_json,
    save_ngboost_model_json,
)
from .ngboost import NGBoost

__all__ = [
    "NGBClassifier",
    "NGBRegressor",
    "NGBSurvival",
    "NGBoost",
    "load_ngboost_model",
    "load_ngboost_model_json",
    "save_ngboost_model_json",
]

__version__ = version(__name__)
