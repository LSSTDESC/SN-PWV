"""Tests for the ``weather`` module"""

from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from astropy.time import Time
from pytz import UTC

from snat_sim import models, time_series_utils
from tests.mock import create_mock_pwv_data


class DatetimeToSecInYear(TestCase):
    """Tests for the ``datetime_to_sec_in_year`` function"""

    def setUp(self):
        """Create a fixed date to test against"""

        self.test_date = datetime(2020, 6, 12)

        # Calculate seconds since january first
        seconds_in_day = 24 * 60 * 60
        time_delta = self.test_date - datetime(2020, 1, 1)
        self.seconds = time_delta.days * seconds_in_day

    def test_seconds_for_jan_1st(self):
        """Test the return is zero for midnight on January 1st"""

        self.assertEqual(time_series_utils.datetime_to_sec_in_year(datetime(2020, 1, 1)), 0)

    def test_seconds_for_known_date(self):
        """Test correct number of seconds are returned for a pre-specified date"""

        returned_seconds = time_series_utils.datetime_to_sec_in_year(self.test_date)
        self.assertEqual(returned_seconds, self.seconds)

    def test_pandas_timestamp_support(self):
        """Test returned values are the same for Timestamp datetime objects"""

        return_for_datetime = time_series_utils.datetime_to_sec_in_year(self.test_date)
        return_timestamp = time_series_utils.datetime_to_sec_in_year(pd.Timestamp(self.test_date))
        self.assertEqual(return_timestamp, return_for_datetime)

    def test_pandas_datetime_index_support(self):
        """Test returned values are the same for DatetimeIndex and list objects"""

        date_as_list = [self.test_date, self.test_date]
        return_for_list = time_series_utils.datetime_to_sec_in_year(date_as_list)
        return_datetime_index = time_series_utils.datetime_to_sec_in_year(pd.DatetimeIndex(date_as_list))
        np.testing.assert_array_equal(return_datetime_index, return_for_list)


class SupplementedData(TestCase):
    """Tests for the ``supplemented_data`` function"""

    @classmethod
    def setUpClass(cls):
        """Create a mock data set for testing"""

        # Create data where every year has an overlapping datapoint with 0, 1,
        # and 2 other years. Also include one data point from a year that
        # should be ignored by the function call.
        index = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4),
                 datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 5), datetime(2021, 1, 6),
                 datetime(2022, 1, 1), datetime(2022, 1, 3), datetime(2022, 1, 5), datetime(2022, 1, 7),
                 datetime(2023, 1, 8)]

        data = np.concatenate([np.zeros(4), np.ones(4), np.full(5, 2)])
        data[2] = np.nan  # We expect data at this point to be overwritten even though it is in the primary year

        cls.input_data = pd.Series(data=data, index=index)
        cls.supplemented = time_series_utils.supplemented_data(cls.input_data, 2020, (2022, 2021))

    def test_primary_year_takes_priority(self):
        """Test entries from the primary Series are kept in favor of the secondary data"""

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 1)], 0)

    def test_priority_maintained_for_supplemental_years(self):
        """Test entries from the secondary Series' follow priority order
        of the ``supp_years`` argument.
        """

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 5)], 2)

    def test_call_with_no_supp_years(self):
        """Test passing no supplementary years returns only the primary year"""

        supplemented = time_series_utils.supplemented_data(self.input_data, 2020)
        self.assertTrue((supplemented.index.year == 2020).all())

    def test_nans_dropped_from_primary(self):
        """Test nan values in the primary year are replaced by supplementary years"""

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 3)], 2)

    def test_unselected_years_are_ignored(self):
        """Test years not passed as arguments are ignored"""

        self.assertNotIn(2023, self.supplemented.index.year)

    def test_raises_error_for_missing_year(self):
        """Test an error is through if a year is passed that is not in the data index"""

        with self.assertRaises(ValueError):
            time_series_utils.supplemented_data(self.input_data, 2020, (2019,))


class PeriodicInterpolation(TestCase):
    """Tests for the ``periodic_interpolation`` function"""

    def setUp(self):
        """Create a series with missing data at the boundaries"""

        self.test_series = pd.Series(np.arange(9, 21))
        self.test_series.iloc[[0, -1]] = np.nan

    def test_warns_if_dtype_is_object(self):
        """A Runtime warning should be raised if the input series data is object dtype"""

        with self.assertWarns(RuntimeWarning):
            time_series_utils.periodic_interpolation(self.test_series.astype(np.dtype('O')))

    def test_boundary_values_are_interpolated(self):
        """Test boundary values are filled using a linear interpolation"""

        interpolated_series = time_series_utils.periodic_interpolation(self.test_series)
        self.assertEqual(interpolated_series.iloc[0], 13)
        self.assertEqual(interpolated_series.iloc[-1], 16)


class ResampleDataAcrossYear(TestCase):
    """Tests for the ``resample_data_across_year`` function"""

    @classmethod
    def setUpClass(cls):
        """Create a series indexed by datetime values"""

        cls.start_time = datetime(2020, 1, 2)
        cls.end_time = datetime(2020, 11, 30)
        cls.delta = timedelta(days=1)
        cls.offset = timedelta(hours=4)

        cls.test_series = create_mock_pwv_data(cls.start_time, cls.end_time, cls.delta, cls.offset)
        cls.resampled_series = time_series_utils.resample_data_across_year(cls.test_series)

    def test_timezone_supported(self):
        """Test the function is timezone aware"""

        # Assigning the timezone before resampling should be the same as
        # assigning it afterword
        test_series_with_tz = self.test_series.copy()
        test_series_with_tz.index = test_series_with_tz.index.tz_localize(UTC)

        resampled_series_with_tz = self.resampled_series.copy()
        resampled_series_with_tz.index = resampled_series_with_tz.index.tz_localize(UTC)

        pd.testing.assert_series_equal(
            resampled_series_with_tz,
            time_series_utils.resample_data_across_year(test_series_with_tz))

    def test_offset_of_returned_index(self):
        """Test returned index has same linear offset as input series"""

        offset = self.resampled_series.index[0] - datetime(2020, 1, 1)
        self.assertEqual(self.offset, offset)

    def test_sampling_rate_of_returned_index(self):
        """Test returned index has same delta value as the input series"""

        delta = self.resampled_series.index[1] - self.resampled_series.index[0]
        self.assertEqual(self.delta, delta)

    def test_returned_date_range(self):
        """Test date range of returned index spans the entire year"""

        expected_start = self.start_time.replace(month=1, day=1) + self.offset
        self.assertEqual(expected_start, self.resampled_series.index.min(), 'Incorrect start value for range.')

        expected_end = expected_start.replace(year=expected_start.year + 1) - self.delta
        self.assertEqual(expected_end, self.resampled_series.index.max(), 'Incorrect end value for range.')

    def test_input_with_zero_offset(self):
        """Test returned offset is zero for input series with zero offset"""

        index = np.arange(datetime(2020, 1, 2), self.end_time, timedelta(days=1)).astype(datetime)
        input_series = pd.Series(np.ones_like(index), index=index)
        resampled_series = time_series_utils.resample_data_across_year(input_series)
        offset = resampled_series.index[0] - datetime(2020, 1, 1)
        self.assertEqual(offset, timedelta(days=0))


class BuildPWVModel(TestCase):
    """Tests for the model returned by the ``build_pwv_model`` function"""

    @classmethod
    def setUpClass(cls):
        """Build a linear PWV interpolation model"""

        cls.test_data = create_mock_pwv_data()
        cls.pwv_model = models.PWVModel(cls.test_data)

    def test_return_matches_input_on_grid_points(self):
        """Test the interpolation function returns the original
        sampled values on the grid points"""

        np.testing.assert_array_equal(
            self.pwv_model.pwv_zenith(self.test_data.index),
            self.test_data.values
        )

    def test_interpolation_between_grid_points(self):
        """Test values between grid points are linearly interpolated"""

        # Pick a point halfway between the first and second grid point
        test_date = self.test_data.index[0] + (self.test_data.index[1] - self.test_data.index[0]) / 2
        expected_value = (self.test_data.iloc[0] + self.test_data.iloc[1]) / 2
        self.assertEqual(self.pwv_model.pwv_zenith(test_date), expected_value)

    def test_return_is_invariant_with_time_format(self):
        """Test changing the time format does not change the returned value"""

        # Pick a non-grid point
        test_date = self.test_data.index[0] + (self.test_data.index[1] - self.test_data.index[0]) / 2
        return_for_datetime = self.pwv_model.pwv_zenith(test_date)
        return_for_mjd = self.pwv_model.pwv_zenith(Time(test_date).to_value('mjd'), time_format='mjd')
        self.assertEqual(return_for_mjd, return_for_datetime)
