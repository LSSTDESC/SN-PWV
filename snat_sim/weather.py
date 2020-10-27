# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""The ``weather`` module is used to characterize atmospheric variability
by modeling the time variable precipitate water vapor (PWV) as a function of
time.

Module API
----------
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy.time import Time


def datetime_to_sec_in_year(dates):
    """Calculate number of seconds elapsed modulo 1 year

    Args:
        dates (array, pd.Datetime): Pandas datetime array

    Returns:
        A numpy array of integers
    """

    dates = pd.to_datetime(dates)

    hour_in_day = 24
    min_in_hour = sec_in_min = 60
    return (
            dates.dayofyear * hour_in_day * min_in_hour * sec_in_min
            + dates.hour * min_in_hour * sec_in_min
            + dates.minute * sec_in_min
    )


def supplemented_data(input_data, primary_year, supp_years, resample_rate='30min'):
    """Return a subset of data for a given year supplemented with data from other years

    Args:
        input_data (pandas.Series): Series of data to use indexed by datetime
        primary_year       (float): Year to supplement data for
        supp_years         (float): Year to supplement data with
        resample_rate        (str): Resample and interpolate supplemented data at the given rate

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
    return stacked_pwv.resample(resample_rate, offset=timedelta(minutes=15)).interpolate()


def index_series_by_seconds(series):
    """Return a copy of a pandas Datetime series indexed by seconds elapsed
    in the year

    Missing values, including those at the beginning and end of the year
    are interpolated for.

    Args:
        series (pd.Series): A series with a Datetime index

    Returns:
        A copy of the passed series with a new index
    """

    # Convert index values to seconds
    series = series.copy()
    series.index = datetime_to_sec_in_year(series.index)

    # Resample the index to span the whole year
    end_of_year = 365.25 * 24 * 60 * 60  # Days in year * hours * min * sec
    delta = series.index[1] - series.index[0]
    offset = series.index[1] % delta
    new_indices = np.arange(-offset, end_of_year + offset + 2 * delta, delta)
    series = series.reindex(new_indices)

    # Wrap values across the boundaries and fill nans with interpolation
    first_not_nan, *_, last_not_nan = np.where(~series.isna())[0]
    series.iloc[0] = series.iloc[last_not_nan]
    series.iloc[-1] = series.iloc[first_not_nan]
    return series.interpolate()


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
