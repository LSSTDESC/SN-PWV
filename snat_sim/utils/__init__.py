"""The ``utils`` module acts as a grab-bag of data processing utilities
Tasks of a similar nature are grouped together into submodules.
"""


def setup_environment() -> None:
    from . import time_series
    from .filters import register_lsst_filters, register_decam_filters

    register_lsst_filters()
    register_decam_filters()
