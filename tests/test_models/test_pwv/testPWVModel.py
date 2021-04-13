"""Tests for the ``snat_sim.modeling.pwv.PWVModel`` class"""

from unittest import TestCase

import numpy as np
from astropy.time import Time

from snat_sim.models import PWVModel
from tests.mock import create_mock_pwv_data


class Interpolation(TestCase):
    """Tests for interpolation of PWV values by the model"""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a linear PWV interpolation model"""

        cls.test_data = create_mock_pwv_data()
        cls.pwv_model = PWVModel(cls.test_data)

    def test_return_matches_input_on_grid_points(self) -> None:
        """Test the interpolation function returns the original
        sampled values on the grid points"""

        np.testing.assert_array_equal(
            self.pwv_model.pwv_zenith(self.test_data.index, time_format=None),
            self.test_data.values
        )

    def test_interpolation_between_grid_points(self) -> None:
        """Test values between grid points are linearly interpolated"""

        # Pick a point halfway between the first and second grid point
        test_date = self.test_data.index[0] + (self.test_data.index[1] - self.test_data.index[0]) / 2
        expected_value = (self.test_data.iloc[0] + self.test_data.iloc[1]) / 2
        self.assertEqual(self.pwv_model.pwv_zenith(test_date, time_format=None), expected_value)

    def test_return_is_invariant_with_time_format(self) -> None:
        """Test changing the time format does not change the returned value"""

        # Pick a non-grid point
        test_date = self.test_data.index[0] + (self.test_data.index[1] - self.test_data.index[0]) / 2
        return_for_datetime = self.pwv_model.pwv_zenith(test_date, time_format=None)
        return_for_mjd = self.pwv_model.pwv_zenith(Time(test_date).to_value('mjd'), time_format='mjd')
        self.assertEqual(return_for_mjd, return_for_datetime)
