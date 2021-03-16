"""The ``time_series`` module extends ``pandas`` functionality
for manipulating time series data. It is intended support tasks particular to
dealing with atmospheric / weather data.

Usage Example
-------------

Importing the ``snat_sim`` package will automatically register methods of
the ``TSUAccessor`` class with the ``pandas`` package. This means the methods
can be called directly on ``pandas.Series`` objects using the ``tsu`` accessor
attribute.

As an example, we create a ``pandas.Series`` object with missing data and
fill in the missing data using the ``periodic_interpolation`` method.

.. doctest:: python

   >>> import numpy as np
   >>> import pandas as pd

   >>> # Create a Series with missing data
   >>> demo_series = pd.Series(np.arange(10, 21))
   >>> demo_series.iloc[[0, -1]] = np.nan
   >>> print(demo_series)
   0      NaN
   1     11.0
   2     12.0
   3     13.0
   4     14.0
   5     15.0
   6     16.0
   7     17.0
   8     18.0
   9     19.0
   10     NaN
   dtype: float64


   >>> # Interpolate for the missing data using periodic boundary conditions
   >>> print(demo_series.tsu.periodic_interpolation())
   0     13.666667
   1     11.000000
   2     12.000000
   3     13.000000
   4     14.000000
   5     15.000000
   6     16.000000
   7     17.000000
   8     18.000000
   9     19.000000
   10    16.333333
   dtype: float64

For information on what other methods are incorporated under the ``tsu``
accessor attribut, see the ``TSUAccessor`` class.

Module Docs
-----------
"""

import datetime as dt
import warnings
from typing import *

import numpy as np
import pandas as pd
from astropy import units as u

from snat_sim import types


def datetime_to_sec_in_year(date: types.DateColl) ->  types.NumpyLike:
    """Calculate number of seconds elapsed modulo 1 year.

    Accurate to within a microsecond.

    Args:
        date: Date(s) to calculate seconds for

    Returns:
        A single float if the input is a single datetime, or a numpy array if the input is a collection.
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


@np.vectorize
def datetime_to_season(date: types.DateColl) -> np.ndarray:
    """Determine the calendar season corresponding to a given datetime

    Seasons are labeled as 'winter', 'spring', 'summer', or 'fall'.

    Args:
        date: Datetime value(s)

    Returns:
        An array of strings
    """

    dummy_year = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        ('winter', (dt.date(dummy_year, 1, 1), dt.date(dummy_year, 3, 20))),
        ('spring', (dt.date(dummy_year, 3, 20), dt.date(dummy_year, 6, 20))),
        ('summer', (dt.date(dummy_year, 6, 20), dt.date(dummy_year, 9, 22))),
        ('fall', (dt.date(dummy_year, 9, 22), dt.date(dummy_year, 12, 20))),
        ('winter', (dt.date(dummy_year, 12, 20), dt.date(dummy_year + 1, 1, 1)))
    ]

    date = date.date().replace(year=dummy_year)
    return cast(np.ndarray, next(season for season, (start, end) in seasons if start <= date < end))


@pd.api.extensions.register_series_accessor('tsu')
class TSUAccessor:
    """Pandas Series accessor for time series utilities"""

    def __init__(self, pandas_obj: pd.Series) -> None:
        """Extends ``pandas`` support for time series data

        DO NOT USE THIS CLASS DIRECTLY! This class is registered as a pandas accessor.
        See the module level usage example for more information.
        """

        self._obj = pandas_obj

    def supplemented_data(self, year: int, supp_years: Collection[int] = tuple()) -> pd.Series:
        """Return the supplemented subset of the series corresponding to a given year

        Data for the given year is supplemented with any available data from
        supplementary years by asserting that the measured values from
        supplementary years are exactly the same as they would be if taken during
        the primary year. Priority is given to supplementary years in the order
        specified by the ``supp_years`` argument.

        Args:
            year: Year to supplement data for
            supp_years: Years to supplement data with when missing from ``year``

        Returns:
            A pandas Series object
        """

        input_data = self._obj.dropna().sort_index()
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

    def periodic_interpolation(self) -> pd.Series:
        """Linearly interpolate the series using periodic boundary conditions

        Similar to the default linear interpolation used by pandas, but
        missing values at the beginning and end of the series are
        interpolated assuming a periodic boundary condition.

        Returns:
            An interpolated copy of the passed series
        """

        if self._obj.dtype is np.dtype('O'):
            warnings.warn('Interpolation may not work for object data types', RuntimeWarning)

        # Identify non-NAN values closest to the edges of the series
        series = self._obj.sort_index()
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

    def resample_data_across_year(self) -> pd.Series:
        """Resample the series evenly from the beginning of the
        earliest year through the end of the latest year.

        Returns:
            A copy of the passed series interpolated for January first through December 31st
        """

        start_time = self._obj.index[0].replace(month=1, day=1, hour=0, minute=0, second=0)
        end_time = self._obj.index[-1].replace(month=12, day=31, hour=23, minute=59, second=59)
        delta = self._obj.index[1] - self._obj.index[0]

        # Modulo operation to determine any linear offset in the temporal sampling
        offset = self._obj.index[0] - start_time
        while offset >= delta:
            offset -= delta

        index_values = np.arange(start_time, end_time, delta).astype(pd.Timestamp) + offset
        new_index = pd.to_datetime(index_values).tz_localize(self._obj.index.tz)
        return self._obj.reindex(new_index)
