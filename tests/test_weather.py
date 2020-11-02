"""Tests for the ``weather`` module"""

from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

from snat_sim import weather


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

        self.assertEqual(weather.datetime_to_sec_in_year(datetime(2020, 1, 1)), 0)

    def test_seconds_for_known_date(self):
        """Test correct number of seconds are returned for a pre-specified date"""

        returned_seconds = weather.datetime_to_sec_in_year(self.test_date)
        self.assertEqual(returned_seconds, self.seconds)

    def test_pandas_support(self):
        """Test returned values are the same when passing pandas and datetime objects"""

        return_for_datetime = weather.datetime_to_sec_in_year(self.test_date)
        return_for_pandas = weather.datetime_to_sec_in_year(pd.to_datetime(self.test_date))
        self.assertEqual(return_for_pandas, return_for_datetime)


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
        cls.supplemented = weather.supplemented_data(cls.input_data, 2020, (2022, 2021))

    def test_primary_year_takes_priority(self):
        """Test entries from the primary Series are kept in favor of the secondary data"""

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 1)], 0)

    def test_priority_maintained_for_supplemental_years(self):
        """Test entries from the secondary Series' follow priority order
        of the ``supp_years`` argument.
        """

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 5)], 2)

    def test_call_with_no_supp_years(self):
        """Test passing no supplimentary years returns only the primary year"""

        supplemented = weather.supplemented_data(self.input_data, 2020)
        self.assertTrue((supplemented.index.year == 2020).all())

    def test_nans_dropped_from_primary(self):
        """Test nan values in the primary year are replaced by supplimentary years"""

        self.assertEqual(self.supplemented.loc[datetime(2020, 1, 3)], 2)

    def test_unselected_years_are_ignored(self):
        """Test years not passed as arguments are ignored"""

        self.assertNotIn(2023, self.supplemented.index.year)


class ResampleDataAcrossYear(TestCase):
    """Tests for the ``resample_data_across_year`` function"""

    @classmethod
    def setUpClass(cls):
        """Create a series indexed by datetime values"""

        cls.start_time = datetime(2020, 1, 2)
        cls.end_time = datetime(2020, 11, 30)
        cls.delta = timedelta(days=1)
        cls.offset = timedelta(hours=4)

        index = np.arange(cls.start_time, cls.end_time, cls.delta).astype(datetime) + cls.offset
        cls.test_series = pd.Series(np.ones_like(index), index=index)
        cls.resampled_series = weather.resample_data_across_year(cls.test_series)

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
        resampled_series = weather.resample_data_across_year(input_series)
        offset = resampled_series.index[0] - datetime(2020, 1, 1)
        self.assertEqual(offset, timedelta(days=0))


class BuildPWVModel(TestCase):
    """Tests for the ``build_pwv_model`` function"""

    def setUpClass(cls):
        """Build a linear PWV interpolation model"""

        pass

    def test_return_matches_input_on_grid_points(self):
        """Test the interpolation function returns the original
        sampled values on the grid points"""

        self.fail()

    def test_interpolation_between_grid_points(self):
        """Test values between grid points are linearly interpolated"""

        self.fail()

    def test_returns_invariant_with_time_format(self):
        """Test changing the time format does not change the returned value"""

        self.fail()
