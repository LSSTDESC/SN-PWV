"""The ``snat_sim`` package was built to support scientific research
efforts by the  Dark Energy Science Collaboration (DESC) into how atmospheric
variability will impact Type Ia Supernovae observed by the Legacy Survey of
Space and Time (LSST).
"""

import warnings

from tables import NaturalNameWarning

from .utils import setup_environment as _setup_environment

__version__ = '0.1.0'
__author__ = 'Dark Energy Science Collaboration'
__maintainer__ = 'Daniel Perrefort'
__license__ = 'GPL 3.0'

warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=NaturalNameWarning)
_setup_environment()
