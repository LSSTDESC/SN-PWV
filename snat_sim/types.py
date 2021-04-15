"""The ``types`` module defines custom typing aliases used to inform PEP-484
style type hints.

For more information on Python type hinting, see Python
Enhancement Protocol 484: https://www.python.org/dev/peps/pep-0484/ .

Typing Aliases
--------------

The value ``N`` in the below table represents **all** available numerical sizes
available from ``numpy`` (e.g., ``np.int8``, ``np.int16``, etc.).

+---------------------+----------------------------------------------------------+
| Alias               | Typing Equivalence                                       |
+=====================+==========================================================+
| ``Numeric``         | ``Union[intN, floatN, uintN, longlong, ulonglong]``      |
+---------------------+----------------------------------------------------------+
| ``NumericalParams`` | ``Dict[str, Numeric]``                                   |
+---------------------+----------------------------------------------------------+
| ``FloatColl``  `    | ``TypeVar(Numeric, Collection[Numeric], np.ndarray)``    |
+---------------------+----------------------------------------------------------+
| ``StrColl``         | ``Union[str, Collection[str]]``                          |
+---------------------+----------------------------------------------------------+
| ``ModelLike``       | ``Union[sncosmo.Model, SNModel]``                        |
+---------------------+----------------------------------------------------------+
| ``NumpyLike``       | ``Union[Numeric, ndarray]``                              |
+---------------------+----------------------------------------------------------+
| ``PathLike``        | ``Union[str, Path]``                                     |
+---------------------+----------------------------------------------------------+
| ``DateColl``        | ``Union[dt.datetime, Collection[dt.datetime]]``          |
+---------------------+----------------------------------------------------------+
"""

import datetime as dt
from pathlib import Path
from typing import Collection, Dict, TypeVar, Union

import numpy as np
import sncosmo

from .models import SNModel

Numeric = Union[
    float, np.float16, np.float32, np.float64, np.float128,
    int, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.longlong, np.ulonglong
]
NumericalParams = Dict[str, Numeric]
FloatColl = TypeVar('FloatColl', Numeric, Collection[Numeric], np.ndarray)
StrColl = Union[str, Collection[str]]

ModelLike = Union[sncosmo.Model, SNModel]
NumpyLike = Union[Numeric, np.ndarray]
PathLike = Union[str, Path]
DateColl = Union[dt.datetime, Collection[dt.datetime]]
