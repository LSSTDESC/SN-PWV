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
        """Test returned values are the same for pandas and datetime objects"""

        return_for_datetime = weather.datetime_to_sec_in_year(self.test_date)
        return_for_pandas = weather.datetime_to_sec_in_year(pd.to_datetime(self.test_date))
        self.assertEqual(return_for_pandas, return_for_datetime)


class ResampleDataAcrossYear(TestCase):
    """Tests for the ``index_series_by_seconds`` function"""

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

# Todo:
# weather.supplemented_data()
# weather.build_pwv_model()
