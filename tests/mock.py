"""Mock objects used when evaluating the test suite"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from snat_sim.models import PWVModel

_index = np.arange(datetime(2020, 1, 1), datetime(2020, 12, 31), timedelta(days=1))
constant_pwv_value = 4
constant_pwv_model = PWVModel(
    pd.Series(np.full(len(_index), constant_pwv_value), index=pd.to_datetime(_index))
)
