"""Tests for the ``OutputDataModel`` class"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim import constants as const
from snat_sim.fitting_pipeline import OutputDataModel


class OutputValueFormatting(TestCase):
    """Test values are added to the output list in the order matching the output header"""

    @classmethod
    def setUpClass(cls):
        """Run a light-curve fit to determine test data"""

        data = sncosmo.load_example_data()
        model = sncosmo.Model('salt2')
        cls.meta = {'SNID': '123'}
        cls.result, cls.fitted_model = sncosmo.fit_lc(
            data, model,
            ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
            bounds={'z': (0.3, 0.7)})  # bounds on parameters (if any)

        cls.formatted_results = OutputDataModel.build_table_entry(cls.meta, cls.fitted_model, cls.result)
        cls.col_names = OutputDataModel.result_table_col_names(cls.fitted_model)

    def test_object_id_position(self):
        """Test position of SNID in output list"""

        position = self.col_names.index('SNID')
        self.assertEqual(self.formatted_results[position], self.meta['SNID'])

    def test_param_value_positions(self):
        """Test position of parameter values in output list"""

        # Get expected index of first parameter from the column names
        # Assume parameter values are contiguous in the array
        first_param = self.fitted_model.param_names[0]
        param_values_start = self.col_names.index(first_param)
        num_parameters = len(self.fitted_model.parameters)

        np.testing.assert_array_equal(
            self.formatted_results[param_values_start: num_parameters + param_values_start],
            self.fitted_model.parameters)

    def test_error_value_positions(self):
        """Test position of parameter errors in output list"""

        # Get expected index of first parameter from the column names
        # Assume parameter error values are contiguous in the array
        first_param = self.fitted_model.param_names[0]
        errors_start = self.col_names.index(first_param + '_err')
        num_parameters = len(self.fitted_model.parameters)

        np.testing.assert_array_equal(
            self.formatted_results[errors_start: num_parameters + errors_start],
            list(self.result.errors.values()))

    def test_chisq_position(self):
        """Test position of chi-squared and degrees of freedom in output list"""

        chisq_index = self.col_names.index('chisq')
        self.assertEqual(self.formatted_results[chisq_index], self.result.chisq)

        dof_index = self.col_names.index('ndof')
        self.assertEqual(self.formatted_results[dof_index], self.result.ndof)

    def test_magnitude_position(self):
        """Test position of magnitude values in output list"""

        mb_index = self.col_names.index('mb')
        mb = self.fitted_model.bandmag('bessellb', 'ab', time=self.fitted_model['t0'])
        self.assertEqual(self.formatted_results[mb_index], mb)

        abs_mag_index = self.col_names.index('abs_mag')
        abs_mag = self.fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo)
        self.assertEqual(self.formatted_results[abs_mag_index], abs_mag)

    def test_output_length_matches_column_names(self):
        """Test the number of output values match the number of columns names"""

        self.assertEqual(
            len(self.formatted_results),
            len(OutputDataModel.result_table_col_names(self.fitted_model))
        )
