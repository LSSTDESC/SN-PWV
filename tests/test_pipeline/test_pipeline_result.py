"""Tests for the ``OutputDataModel`` class"""

from unittest import TestCase

import numpy as np
import sncosmo

from snat_sim import constants as const
from snat_sim.pipeline import DataModel


class OutputValueFormatting(TestCase):
    """Test values are added to the output list in the order matching the output header"""

    @classmethod
    def setUpClass(cls):
        """Run a light-curve fit and format results as a table entry"""

        data = sncosmo.load_example_data()
        model = sncosmo.Model('salt2')
        cls.meta = {'SNID': '123'}
        cls.result, cls.fitted_model = sncosmo.fit_lc(
            data, model,
            ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
            bounds={'z': (0.3, 0.7)})  # bounds on parameters (if any)

        cls.data_model = DataModel(model, model)
        cls.formatted_results = cls.data_model.build_table_entry(cls.meta, cls.fitted_model, cls.result)

    def test_object_id_position(self):
        """Test position of SNID in output list"""

        position = self.data_model.column_names.index('SNID')
        self.assertEqual(self.formatted_results[position], self.meta['SNID'])

    def test_param_value_positions(self):
        """Test position of parameter values in output list"""

        # Get expected index of first parameter from the column names
        # Assume parameter values are contiguous in the array
        first_param = self.fitted_model.param_names[0]
        param_values_start = self.data_model.column_names.index(first_param)
        num_parameters = len(self.fitted_model.parameters)

        np.testing.assert_array_equal(
            self.formatted_results[param_values_start: num_parameters + param_values_start],
            self.fitted_model.parameters)

    def test_error_value_positions(self):
        """Test position of parameter errors in output list"""

        # Get expected index of first parameter from the column names
        # Assume parameter error values are contiguous in the array
        first_param = self.fitted_model.param_names[0]
        errors_start = self.data_model.column_names.index(first_param + '_err')
        num_parameters = len(self.fitted_model.parameters)

        np.testing.assert_array_equal(
            self.formatted_results[errors_start: num_parameters + errors_start],
            list(self.result.errors.values()))

    def test_chisq_position(self):
        """Test position of chi-squared and degrees of freedom in output list"""

        chisq_index = self.data_model.column_names.index('chisq')
        self.assertEqual(self.formatted_results[chisq_index], self.result.chisq)

        dof_index = self.data_model.column_names.index('ndof')
        self.assertEqual(self.formatted_results[dof_index], self.result.ndof)

    def test_magnitude_position(self):
        """Test position of magnitude values in output list"""

        mb_index = self.data_model.column_names.index('mb')
        mb = self.fitted_model.bandmag('bessellb', 'ab', time=self.fitted_model['t0'])
        self.assertEqual(self.formatted_results[mb_index], mb)

        abs_mag_index = self.data_model.column_names.index('abs_mag')
        abs_mag = self.fitted_model.source_peakabsmag('bessellb', 'ab', cosmo=const.betoule_cosmo)
        self.assertEqual(self.formatted_results[abs_mag_index], abs_mag)

    def test_output_length_matches_column_names(self):
        """Test the number of output values match the number of columns names"""

        self.assertEqual(
            len(self.formatted_results),
            len(self.data_model.column_names)
        )


class MaskedRowCreation(TestCase):
    """Tests for the creation of masked table entries"""

    @classmethod
    def setUpClass(cls):
        """Create a masked table entry"""

        model = sncosmo.Model('salt2')
        cls.meta = {'SNID': '123'}
        cls.fit_failure_exception = ValueError('This fit failed to converge.')

        cls.data_model = DataModel(model, model)
        cls.masked_row = cls.data_model.mask_failed_lc_fit(cls.meta, cls.fit_failure_exception)

    def test_mask_value_is_neg_99(self):
        """Test -99 is used to represent masked values"""

        np.testing.assert_array_equal(self.masked_row[1:-2], -99)

    def test_object_id_position(self):
        """Test position of SNID in output list"""

        snid_index = self.data_model.column_names.index('SNID')
        self.assertEqual(self.masked_row[snid_index], self.meta['SNID'])

    def test_failure_message_position(self):
        """Test position of failure message in output list"""

        message_index = self.data_model.column_names.index('message')
        self.assertEqual(self.masked_row[message_index], str(self.fit_failure_exception))
