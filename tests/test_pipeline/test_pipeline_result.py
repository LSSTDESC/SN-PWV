"""Tests for the ``OutputDataModel`` class"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim.models import SNModel
from snat_sim.pipeline import PipelineResult


class ListFormatting(TestCase):
    """Test values are added to the output list in the order matching the output header"""

    @classmethod
    def setUpClass(cls) -> None:
        """Run a light-curve fit and format results as a table entry"""

        # Build dictionaries of mock s parameter values
        # Ensure the values are all a little different
        demo_params = sncosmo.load_example_data().meta
        fit_params = {k: v * 1.1 for k, v in demo_params.items()}
        fit_errors = {k: v * .1 for k, v in fit_params.items()}
        cls.result = PipelineResult(
            'snid_value', demo_params, fit_params, fit_errors,
            chisq=1.5, ndof=6, mb=6, abs_mag=-19.5, message='Exit Message')

        cls.param_names = list(demo_params.keys())
        cls.result_list = cls.result.to_list(cls.param_names, cls.param_names)
        cls.column_names = cls.result.column_names(cls.param_names, cls.param_names)

    def test_object_id_position(self) -> None:
        """Test position of SNID in output list"""

        position = self.column_names.index('snid')
        self.assertEqual(self.result.snid, self.result_list[position])

    def test_simulated_param_positions(self) -> None:
        """Test position of simulated parameter values in output list"""

        # Get expected index of first parameter from the column names
        # Assume parameter values are contiguous in the array
        first_param = self.param_names[0]
        param_values_start = self.column_names.index('sim_' + first_param)
        num_parameters = len(self.param_names)

        np.testing.assert_array_equal(
            list(self.result.sim_params.values()),
            self.result_list[param_values_start: num_parameters + param_values_start]
        )

    def test_fitted_param_positions(self) -> None:
        """Test position of fitted parameter values in output list"""

        first_param = self.param_names[0]
        param_values_start = self.column_names.index('fit_' + first_param)
        num_parameters = len(self.param_names)

        np.testing.assert_array_equal(
            list(self.result.fit_params.values()),
            self.result_list[param_values_start: num_parameters + param_values_start]
        )

    def test_error_value_positions(self) -> None:
        """Test position of parameter errors in output list"""

        first_param = self.param_names[0]
        param_values_start = self.column_names.index('err_' + first_param)
        num_parameters = len(self.param_names)

        np.testing.assert_array_equal(
            list(self.result.fit_err.values()),
            self.result_list[param_values_start: num_parameters + param_values_start]
        )

    def test_chisq_position(self) -> None:
        """Test position of chi-squared and degrees of freedom in output list"""

        chisq_index = self.column_names.index('chisq')
        self.assertEqual(self.result.chisq, self.result_list[chisq_index])

        dof_index = self.column_names.index('ndof')
        self.assertEqual(self.result.ndof, self.result_list[dof_index])

    def test_magnitude_position(self) -> None:
        """Test position of magnitude values in output list"""

        mb_index = self.column_names.index('mb')
        self.assertEqual(self.result.mb, self.result_list[mb_index])

        abs_mag_index = self.column_names.index('abs_mag')
        self.assertEqual(self.result.abs_mag, self.result_list[abs_mag_index])

    def test_output_length_matches_column_names(self) -> None:
        """Test the number of output values match the number of columns names"""

        self.assertEqual(len(self.column_names), len(self.result_list))


class MaskedRowCreation(TestCase):
    """Tests for the creation of masked table entries"""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a masked table entry"""

        cls.param_names = SNModel('salt2').param_names
        cls.result = PipelineResult('snid_value', message='A status message')
        cls.result_list = cls.result.to_list(cls.param_names, cls.param_names)

    def test_mask_value_is_neg_99(self) -> None:
        """Test -99.99 is used to represent masked values"""

        np.testing.assert_array_equal(self.result_list[1:-2], -99.99)

    def test_unmasked_snid(self) -> None:
        """Test the SNID value is not masked"""

        self.assertEqual(self.result.snid, self.result_list[0])

    def test_unmasked_message(self) -> None:
        """Test the failure message is not masked"""

        self.assertEqual(self.result.message, self.result_list[-1])


class ToCsv(TestCase):
    """Test the conversion of ``PipelineResult`` instances to a CSV string"""

    def test_ends_with_new_line(self) -> None:
        """Test the returned string ends with a newline character"""

        string = PipelineResult('snid_value', message='A status message').to_csv([], [])
        self.assertEqual('\n', string[-1])
