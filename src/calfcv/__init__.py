"""
The Calf and CalfCV classifiers.

===============================================================
Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
License: 3-clause BSD
===============================================================
"""

from ._version import __version__
from .auc_sorter import AUCSorter
from .calf import Calf
from .calfcv import CalfCV

__all__ = ["Calf", "CalfCV", "AUCSorter", "__version__"]
