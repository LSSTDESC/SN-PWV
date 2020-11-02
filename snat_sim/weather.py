# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``weather`` module is used to characterize atmospheric variability
by modeling the time variable precipitate water vapor (PWV) as a function of
time.

Module API
----------
"""

import warnings
from datetime import datetime, timedelta

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


def supplemented_data(input_data, primary_year, supp_years, resample_rate='30min', offset=timedelta(minutes=15)):
    """Return a subset of data for a given year supplemented with data from other years

    Default values for the ``resample`` and ``offset`` arguments are chosen to
    reflect

    Args:
        input_data (pandas.Series): Series of data to use indexed by datetime
        primary_year       (float): Year to supplement data for
        supp_years         (float): Year to supplement data with
        resample_rate        (str): Resample and interpolate supplemented data at the given rate
        offset         (timedelta): Linear offset applied to the resampled index

    Returns:
        A pandas Series object
    """

    years = np.array([primary_year, *supp_years])

    # Check for years with no available data
    missing_years = years[~np.isin(years, input_data.index.year)]
    if missing_years:
        warnings.warn(f'No data for years: {missing_years}')

    # Keep only data for the given hears
    stacked_pwv = input_data[np.isin(input_data.index.year, years)]
    stacked_pwv = stacked_pwv.sort_index()

    # Mak all dates have the same year
    new_index = []
    for date_idx in stacked_pwv.index:
        new_index.append(date_idx.replace(year=years[0]))

    # Keep only unique dates
    stacked_pwv.index = new_index
    stacked_pwv = stacked_pwv[~stacked_pwv.index.duplicated(keep='first')]

    # Resample and interpolate any missing values
    return stacked_pwv.resample(resample_rate, offset=offset).interpolate()


def resample_data_across_year(series):
    """Return a copy of a pandas Datetime series resampled evenly from the
    beginning to the end of the year

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
    while offset > delta:
        offset -= delta

    new_indices = np.arange(start_time, end_time, delta).astype(datetime) + offset
    return series.reindex(new_indices).interpolate()


def build_pwv_model(pwv_series):
    """Build interpolator for the PWV at a given point of the year

    Args:
        pwv_series (Series): PWV values with a datetime index

    Returns:
        An interpolation function that accepts MJD
    """

    pwv_model_data = index_series_by_seconds(pwv_series)

    def interp(mjd):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            x_as_datetime = Time(mjd, format='mjd').to_datetime()
            x_in_seconds = datetime_to_sec_in_year(x_as_datetime)

        return np.interp(
            x=x_in_seconds,
            xp=pwv_model_data.index,
            fp=pwv_model_data.values
        )

    return interp
