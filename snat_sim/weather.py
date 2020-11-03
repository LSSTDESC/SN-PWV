"""The ``weather`` module is used to characterize atmospheric variability
by modeling the time variable precipitate water vapor (PWV) as a function of
time. It also provides limited functionality for manipulating temporal data as
required when building a time based model. Models are based on the linear
interpolation of measured time series data while using periodic boundary
conditions to support modeling values outside the measured time range. In
doing so we assume a level uniformity in the seasonal variation of weather
data over time.

Module API
----------
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time


def datetime_to_sec_in_year(date):
    """Calculate number of seconds elapsed modulo 1 year

    Accurate to within a microsecond

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
    """Return a subset of a dataframe corresponding to a given year

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

    # Identify non-NAN values closest to the edges of the series
    series = series.sort_index()
    delta = series.index[1] - series.index[0]
    start_idx, end_idx = series.iloc[[0, -1]].index
    first_not_nan, last_not_nan = series.dropna().iloc[[0, -1]]

    # Extend the series with temporary values so we can interpolate any missing values
    series.loc[start_idx - 2 * delta] = last_not_nan
    series.loc[start_idx - delta] = np.nan
    series.loc[end_idx + delta] = np.nan
    series.loc[end_idx + 2 * delta] = first_not_nan

    # Drop the temporary values
    return series.sort_index().interpolate().truncate(start_idx, end_idx)


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
    return periodic_interpolation(series.reindex(new_indices))


def build_pwv_model(pwv_series):
    """Build interpolator for the PWV at a given point of the year

    Returned interpolator defaults to expecting datetime in MJD format.

    Args:
        pwv_series (Series): PWV values with a datetime index

    Returns:
        An interpolation function that accepts ``date`` and ``format`` arguments
    """

    # Debugging note: The return of resample_data_across_year should not have
    # NANs, but when running the test suite the somehow get through.
    # See test BuildPWVModel.test_return_matches_input_on_grid_points
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
