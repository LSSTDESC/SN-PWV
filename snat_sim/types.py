from typing import Collection, Dict, TypeVar, Union

import numpy as np

Numeric = Union[
    float, np.float16, np.float32, np.float64, np.float128,
    int, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.longlong, np.ulonglong
]

ModelParams = Dict[str, Numeric]
FloatOrArray = TypeVar('FloatOrArray', Numeric, Collection[Numeric], np.ndarray)
