# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``weather`` module is used to characterize atmospheric variability
by modeling the time variable precipitate water vapor (PWV) as a function of
time.

Module API
----------
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time


def datetime_to_sec_in_year(dates):
    """Calculate number of seconds elapsed modulo 1 year

    Accurate to within a microsecond

    Args:
        dates (datetime, array, pd.Datetime): Pandas datetime array

    Returns:
        A single float or a numpy array of integers
    """

    dates = pd.to_datetime(dates)
    return (
            (dates.dayofyear - 1) * u.day +
            dates.hour * u.hour +
            dates.second * u.s +
            dates.microsecond * u.ms
    ).to(u.s).value


def supplemented_data(input_data, year, supp_years=tuple()):
    """Return a subset of a dataframe corresponding to a given year

    Missing data for the given year is supplemented using data from
    supplementary years by asserting that the measured values from
    supplementary years are exactly the same as they would be if taken during
    the primary year. Priority is given to supplementary years in the order
    specified by the ``supp_years`` argument.

    Args:
        input_data     (pandas.Series): Series of data to use indexed by datetime
        year                   (float): Year to supplement data for
        supp_years (collection[float]): Years to supplement data with when missing from ``year``

    Returns:
        A pandas Series object
    """

    input_data = input_data.dropna().sort_index()
    years = np.array([year, *supp_years])

    # Check for years with no available data
    missing_years = years[~np.isin(years, input_data.index.year)]
    if missing_years:
        raise ValueError(f'No data for years: {missing_years}')

    # Keep only data for the given years while maintaining priority order
    stacked_pwv = pd.concat(
        [input_data[input_data.index.year == yr] for yr in years]
    )

    # Make all dates have the same year and keep only unique dates
    stacked_pwv.index = [date_idx.replace(year=years[0]) for date_idx in stacked_pwv.index]
    return stacked_pwv[~stacked_pwv.index.duplicated(keep='first')]


def resample_data_across_year(series):
    """Return a copy of a pandas Datetime series resampled evenly from the
    beginning of the earliest year through the end of the latest year.

    Args:
        series (pd.Series): A series with a Datetime index

    Returns:
        A copy of the passed series interpolated for January first through December 31st
    """

    start_time = series.index[0].replace(month=1, day=1, hour=0, second=0)
    end_time = series.index[-1].replace(month=12, day=31, hour=23, minute=59, second=59)
    delta = series.index[1] - series.index[0]

    # Modulo operation to determine any linear offset in the temporal sampling
    offset = series.index[0] - start_time
    while offset >= delta:
        offset -= delta

    new_indices = np.arange(start_time, end_time, delta).astype(datetime) + offset
    return series.reindex(new_indices).interpolate()


def build_pwv_model(pwv_series):
    """Build interpolator for the PWV at a given point of the year

    Returned interpolator defaults to expecting datetime in MJD format.

    Args:
        pwv_series (Series): PWV values with a datetime index

    Returns:
        An interpolation function that accepts ``date`` and ``format`` arguments
    """

    pwv_model_data = resample_data_across_year(pwv_series)
    pwv_model_data.index = datetime_to_sec_in_year(pwv_model_data.index)

    def interp_pwv(date, format='mjd'):
        f"""Interpolate the PWV as a function of time
        
        Args:
            date (float): The date to interpolate PWV for
            format (str): Astropy supported time format of the ``date`` argument
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_as_datetime = Time(date, format=format).to_datetime()
            x_in_seconds = datetime_to_sec_in_year(x_as_datetime)

        return np.interp(
            x=x_in_seconds,
            xp=pwv_model_data.index,
            fp=pwv_model_data.values
        )

    return interp_pwv


def build_suominet_model(receiver, year, supp_years):
    """Similar to the ``build_pwv_model`` function, but automatically builds a
    model from a data taken by a SuomiNet GPS receiver

    Args:
        receiver (pwv_kpno.GPSReceiver): GPS receiver to access data from
        year                    (float): Year to use data from when building the model
        supp_years              (float): Years to supplement data with when missing from ``year``

    Returns:
        An interpolation function that accepts ``date`` and ``format`` arguments
    """

    weather_data = receiver.weather_data().PWV
    supp_data = supplemented_data(weather_data, year, supp_years)
    resampled_data = resample_data_across_year(supp_data)
    return build_pwv_model(resampled_data)
