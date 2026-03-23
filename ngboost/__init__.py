"""The NGBoost Library"""

try:
    from importlib.metadata import version
except ImportError:
    # before python 3.8
    from importlib_metadata import version

from .api import NGBClassifier, NGBRegressor, NGBSurvival
from .helpers import load_ngboost_model
from .ngboost import NGBoost

__all__ = [
    "NGBClassifier",
    "NGBRegressor",
    "NGBSurvival",
    "NGBoost",
    "load_ngboost_model",
]

__version__ = version(__name__)
