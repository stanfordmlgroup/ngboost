"""The NGBoost Library"""

try:
    from importlib.metadata import version
except ImportError:
    # before python 3.8
    from importlib_metadata import version

from .api import NGBClassifier, NGBRegressor, NGBSurvival
from .ngboost import NGBoost

__all__ = ["NGBClassifier", "NGBRegressor", "NGBSurvival", "NGBoost"]

__version__ = version(__name__)
