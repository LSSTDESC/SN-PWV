"""Tests for the ``weather`` module"""

from datetime import datetime
from unittest import TestCase

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
        """Test returned val;ues are the same for pandas and datetime objects"""

        return_for_datetime = weather.datetime_to_sec_in_year(self.test_date)
        return_for_pandas = weather.datetime_to_sec_in_year(pd.to_datetime(self.test_date))
        self.assertEqual(return_for_pandas, return_for_datetime)


class IndexSeriesBySeconds(TestCase):
    """Tests for the ``index_series_by_seconds`` function"""

    def setUp(self):
        """Create a series indexed by datetime values"""

        data = range(1, 11)
        datetimes = [datetime(2020, 1, d) for d in data]
        self.test_series = pd.Series(data, index=datetimes)

    def test_returned_index_is_in_seconds(self):
        """Compare index of returned series against return from ``datetime_to_sec_in_year``"""

        returned_series = weather.index_series_by_seconds(self.test_series)
        expected_index = weather.datetime_to_sec_in_year(self.test_series.index)
        pd.testing.assert_series_equal(returned_series.index, expected_index)

# Todo:
# weather.supplemented_data()
# weather.build_pwv_model()
