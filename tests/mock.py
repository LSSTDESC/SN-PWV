"""Mock objects used when evaluating the test suite"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.table import Table

from snat_sim.models import PWVModel


def create_mock_pwv_data(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 12, 31),
        delta=timedelta(days=1),
        offset=timedelta(hours=0)):
    """Return a ``Series`` of mock PWV values that alternate between .5 and 1

    Args:
        start_time (datetime): Start date of the returned series index
        end_time   (datetime): End date of the returned series index
        delta     (timedelta): Sampling rate of the datetime values
        offset    (timedelta): Apply a linear offset to the returned values

    Returns:
        A pandas ``Series`` object
    """

    index = np.arange(start_time, end_time, delta).astype(datetime) + offset
    pwv = np.ones_like(index, dtype=float)
    pwv[::2] = 0.5
    return pd.Series(pwv, index=index)


def create_constant_pwv_model(constant_pwv_value=4):
    """Create a ``PWVModel`` instance that returns a constant PWV at zenith

    Args:
        constant_pwv_value (float): The PWV value to use as the constant

    Returns:
        A ``PWVModel`` instance
    """

    date_sampling = np.arange(datetime(2020, 1, 1), datetime(2020, 12, 31), timedelta(days=1))
    pwv = np.full(len(date_sampling), constant_pwv_value)
    model_data = pd.Series(pwv, index=pd.to_datetime(date_sampling))
    return PWVModel(model_data)


def create_mock_plasticc_light_curve():
    """Create a mock light-curve in the PLaSTICC data format

    Returns:
        An astropy table
    """

    time_values = np.arange(-20, 52)
    return Table(
        data={
            'MJD': time_values,
            'FLT': list('ugrizY') * (len(time_values) // 6),
            'FLUXCAL': np.ones_like(time_values),
            'FLUXCALERR': np.full_like(time_values, .2),
            'ZEROPT': np.full_like(time_values, 30),
            'PHOTFLAG': [0] * 10 + [6144] + [4096] * 61,
            'SKY_SIG': np.full_like(time_values, 80)
        },
        meta={
            'SIM_PEAKMJD': 0,
            'SIM_SALT2x1': .1,
            'SIM_SALT2c': .2,
            'SIM_REDSHIFT_CMB': .5,
            'SIM_SALT2x0': 1
        }
    )
