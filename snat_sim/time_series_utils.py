"""The ``time_series_utils.py`` module is used provides limited functionality
for manipulating time series data. It is intended to supplement existing
functionality in the ``pandas`` package with support for tasks particular to
dealing with atmospheric / weather data.

Module Docs
-----------
"""

import datetime
import warnings

import numpy as np
import pandas as pd
from astropy import units as u


def datetime_to_sec_in_year(date):
    """Calculate number of seconds elapsed modulo 1 year

    Accurate to within a microsecond.

    Args:
        date (datetime, array, pd.Datetime): Pandas datetime array

    Returns:
        A single float or a numpy array of integers
    """

    # Using ``atleast_1d`` with ``to_datetime`` guarantees a ``DatetimeIndex``
    # object is returned, otherwise we get a ``TimeStamp`` object for scalars
    # which has different attributes names than the ones we use below
    pandas_dates = pd.to_datetime(np.atleast_1d(date))

    # The ``values`` attributes returns a numpy array. Pandas objects
    # are not generically compatible with astropy units
    seconds = (
            (pandas_dates.dayofyear.values - 1) * u.day +
            pandas_dates.hour.values * u.hour +
            pandas_dates.second.values * u.s +
            pandas_dates.microsecond.values * u.ms
    ).to(u.s).value

    # If the argument was a scalar, return a scalar
    if np.ndim(date) == 0:
        seconds = np.asscalar(seconds)

    return seconds


def supplemented_data(input_data, year, supp_years=tuple()):
    """Return the supplemented subset of a dataframe corresponding to a given year

    Data for the given year is supplemented with any available data from
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


def periodic_interpolation(series):
    """Similar to linear interpolation on a pandas array, but missing values
    at the beginning and end of the series are interpolated assuming a periodic
    boundary condition.

    Args:
        series (pandas.Series): A Pandas Series to infill by linear interpolation

    Returns:
        An interpolated copy of the passed series
    """

    if series.dtype is np.dtype('O'):
        warnings.warn('Interpolation may not work for object data types', RuntimeWarning)

    # Identify non-NAN values closest to the edges of the series
    series = series.sort_index()
    delta = series.index[1] - series.index[0]
    start_idx, end_idx = series.iloc[[0, -1]].index
    first_not_nan, last_not_nan = series.dropna().iloc[[0, -1]]

    # Extend the series with temporary values so we can interpolate any missing values
    series.loc[start_idx - 2 * delta] = last_not_nan
    series.loc[start_idx - delta] = np.NAN
    series.loc[end_idx + delta] = np.NAN
    series.loc[end_idx + 2 * delta] = first_not_nan

    # Drop the temporary values
    return series.sort_index().interpolate().truncate(start_idx, end_idx)


def resample_data_across_year(series):
    """Return a copy of a pandas series resampled evenly from the
    beginning of the earliest year through the end of the latest year.

    Args:
        series (pd.Series): A series with a Datetime index

    Returns:
        A copy of the passed series interpolated for January first through December 31st
    """

    start_time = series.index[0].replace(month=1, day=1, hour=0, minute=0, second=0)
    end_time = series.index[-1].replace(month=12, day=31, hour=23, minute=59, second=59)
    delta = series.index[1] - series.index[0]

    # Modulo operation to determine any linear offset in the temporal sampling
    offset = series.index[0] - start_time
    while offset >= delta:
        offset -= delta

    index_values = np.arange(start_time, end_time, delta).astype(pd.Timestamp) + offset
    new_index = pd.to_datetime(index_values).tz_localize(series.index.tz)
    return series.reindex(new_index)


@np.vectorize
def datetime_to_season(time):
    """Determine the calendar season corresponding to a given datetime

    Seasons are labeled as 'winter', 'spring', 'summer', or 'fall'.

    Args:
        time (Datetime, List[Datetime]): Datetime value(s)

    Returns:
        An array of strings.
    """
    dummy_year = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        ('winter', (datetime.date(dummy_year, 1, 1), datetime.date(dummy_year, 3, 20))),
        ('spring', (datetime.date(dummy_year, 3, 21), datetime.date(dummy_year, 6, 20))),
        ('summer', (datetime.date(dummy_year, 6, 21), datetime.date(dummy_year, 9, 22))),
        ('fall', (datetime.date(dummy_year, 9, 23), datetime.date(dummy_year, 12, 20))),
        ('winter', (datetime.date(dummy_year, 12, 21), datetime.date(dummy_year, 12, 31)))
    ]

    time = time.datetime.date().replace(year=dummy_year)
    return next(season for season, (start, end) in seasons if start <= time <= end)
