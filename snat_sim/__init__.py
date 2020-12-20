"""The ``snat_sim`` package was built to support scientific research
efforts by the  Dark Energy Science Collaboration (DESC) into how atmospheric
variability will impact Type Ia Supernovae observed by the Legacy Survey of
Space and Time (LSST).
"""

from . import *

# Register pandas accessors
from .utils.time_series import TSUAccessor

__version__ = 'Development'
__author__ = 'Dark Energy Science Collaboration'
__maintainer__ = 'Daniel Perrefort'
__license__ = 'GPL 3.0'
