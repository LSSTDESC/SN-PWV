from unittest import TestCase

import numpy as np
import pandas as pd
import sncosmo

from snat_sim.models import SNModel
from snat_sim.pipeline.data_model import MASK_VALUE, PipelinePacket


class Base:
    """Generic setup tasks for testing PipelinePacket objects"""

    @classmethod
    def setUpClass(cls) -> None:
        """Run a light-curve fit and format results as a table entry"""

        cls.data = sncosmo.load_example_data()
        cls.model_to_fit = SNModel('Salt2')
        cls.model_to_fit.update(cls.data.meta)

        cls.params_to_fit = ['x1', 'c']
        fit_result, fitted_model = cls.model_to_fit.fit_lc(cls.data, cls.params_to_fit)
        cls.packet = PipelinePacket(
            snid=1234,
            sim_params=cls.data.meta,
            light_curve=cls.data,
            fit_result=fit_result,
            fitted_model=fitted_model,
        )


class SimParamsToPandas(Base, TestCase):
    """Test the casting of simulated parameters to a pandas object"""

    def test_return_is_dataframe(self) -> None:
        """Test the returned object is a dataframe"""

        test_data = self.packet.sim_params_to_pandas()
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(1, len(test_data))

    def test_colnames_match_param_names(self) -> None:
        """Test the column names in the returned dataframe match simulated parameters"""

        sim_params = list(self.packet.sim_params.keys()) + ['SNID']
        columns = self.packet.sim_params_to_pandas().columns
        np.testing.assert_array_equal(sim_params, columns)

    def test_values_match_params(self) -> None:
        """Test the values of the returned dataframe match simulated parameters"""

        returned_param_vals = dict(self.packet.sim_params_to_pandas().iloc[0])
        expected_vals = dict(self.packet.sim_params)
        expected_vals['SNID'] = self.packet.snid

        self.assertDictEqual(returned_param_vals, expected_vals)


class FittedParamsToPandas(Base, TestCase):
    """Test the casting of fitted parameters to a pandas object"""

    def test_return_is_dataframe(self) -> None:
        """Test the returned object is a dataframe"""

        test_data = self.packet.fitted_params_to_pandas()
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(1, len(test_data))

    def test_colnames_match_param_names(self) -> None:
        """Test the column names in the returned dataframe match simulated parameters"""

        fit_params = ['snid']
        fit_params.extend('fit_' + param for param in self.model_to_fit.param_names)
        fit_params.extend('err_' + param for param in self.model_to_fit.param_names)
        fit_params.extend(('chisq', 'ndof', 'mb', 'abs_mag', 'message'))
        columns = self.packet.fitted_params_to_pandas().columns
        np.testing.assert_array_equal(fit_params, columns)

    def test_values_match_params(self) -> None:
        """Test the values of the returned dataframe match simulated parameters"""

        returned_data = self.packet.fitted_params_to_pandas().iloc[0]

        self.assertEqual(self.packet.snid, returned_data['snid'], 'Incorrect SNID')
        self.assertEqual(self.packet.fit_result.chisq, returned_data['chisq'], 'Incorrect chisq')
        self.assertEqual(self.packet.fit_result.ndof, returned_data['ndof'], 'Incorrect ndof')
        self.assertEqual(self.packet.message, returned_data['message'], 'Incorrect result message')

        for param in self.model_to_fit.param_names:
            fit_result = self.packet.fitted_model[param]
            return_val = returned_data[f'fit_{param}']
            self.assertEqual(fit_result, return_val, f'Incorrect fit value for {param}')

            fit_err = self.packet.fit_result.errors.get(param, MASK_VALUE)
            return_err = returned_data[f'err_{param}']
            self.assertEqual(fit_err, return_err, f'Incorrect error value for {param}')

    def test_values_are_masked_for_no_fit(self) -> None:
        """Test returned values are masked when no fit results are stored in the packet"""

        # Create a packet with no fit results.  
        packet = PipelinePacket(snid=1234, sim_params=self.data.meta, light_curve=self.data, message='dummy message')
        masked_values = packet.fitted_params_to_pandas().iloc[0].values

        # All values should be masked other than the SNID and message
        self.assertEqual(packet.snid, masked_values[0])
        np.testing.assert_array_equal(MASK_VALUE, masked_values[1:-1])
        self.assertEqual(packet.message, masked_values[-1])
