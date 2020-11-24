"""Tests for the ``OutputDataModel`` class"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim.fitting_pipeline import OutputDataModel


class OutputValueFormatting(TestCase):

    @classmethod
    def setUpClass(cls):
        data = sncosmo.load_example_data()
        model = sncosmo.Model('salt2')
        cls.meta = {'SNID': '123'}
        cls.result, cls.fitted_model = sncosmo.fit_lc(
            data, model,
            ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
            bounds={'z': (0.3, 0.7)})  # bounds on parameters (if any)

        cls.formatted_results = OutputDataModel.build_result_table_entry(cls.meta, cls.fitted_model, cls.result)

    def test_value_positions_in_output_list(self):
        """Test values are added to the output list in the order matching the output header"""

        self.assertEqual(self.formatted_results[0], self.meta['SNID'])

        param_start = 1
        num_parameters = len(self.fitted_model.parameters)
        np.testing.assert_array_equal(
            self.formatted_results[param_start: num_parameters + param_start],
            self.fitted_model.parameters)

        errors_start = num_parameters + param_start
        errors = list(self.result.errors.values())
        np.testing.assert_array_equal(
            self.formatted_results[errors_start: num_parameters + errors_start],
            errors)

        chisq_start = errors_start + len(errors)
        np.testing.assert_array_equal(
            self.formatted_results[chisq_start: chisq_start + 2],
            [self.result.chisq, self.result.ndof])

    def test_output_length_matches_column_names(self):
        self.assertEqual(
            len(self.formatted_results),
            len(OutputDataModel.result_table_col_names(self.fitted_model))
        )
