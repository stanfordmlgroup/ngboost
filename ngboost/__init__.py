"""The NGBoost Library"""

try:
    from importlib.metadata import version
except ImportError:
    # before python 3.8
    from importlib_metadata import version

from .api import NGBClassifier, NGBRegressor, NGBSurvival  # NOQA
from .ngboost import NGBoost  # NOQA

__version__ = version(__name__)
