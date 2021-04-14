"""Tests for the ``snat_sim.pipeline.data_model.PipelinePacket`` class"""

from unittest import TestCase

import numpy as np
import pandas as pd

from snat_sim.pipeline.data_model import MASK_VALUE
from tests.mock import create_mock_pipeline_packet


class SimParamsToPandas(TestCase):
    """Test the casting of simulated parameters to a pandas object"""

    def setUp(self) -> None:
        """Create a mock pipeline packet for testing"""

        self.packet = create_mock_pipeline_packet()

    def test_return_is_dataframe(self) -> None:
        """Test the returned object is a dataframe"""

        test_data = self.packet.sim_params_to_pandas()
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(1, len(test_data))

    def test_colnames_match_param_names(self) -> None:
        """Test the column names in the returned dataframe match simulated parameters"""

        sim_params = list(self.packet.sim_params.keys())
        columns = self.packet.sim_params_to_pandas().columns
        np.testing.assert_array_equal(sim_params, columns)

    def test_values_match_params(self) -> None:
        """Test the values of the returned dataframe match simulated parameters"""

        returned_params = self.packet.sim_params_to_pandas().iloc[0]
        for key, val in self.packet.sim_params.items():
            self.assertEqual(val, returned_params[key])


class FittedParamsToPandas(TestCase):
    """Test the casting of fitted parameters to a pandas object"""

    def setUp(self) -> None:
        """Create a mock pipeline packet for testing"""

        self.packet = create_mock_pipeline_packet()

    def test_return_is_dataframe(self) -> None:
        """Test the returned object is a dataframe"""

        test_data = self.packet.fitted_params_to_pandas()
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(1, len(test_data))

    def test_colnames_match_param_names(self) -> None:
        """Test the column names in the returned dataframe match simulated parameters"""

        fit_params = ['snid']
        fit_params.extend('fit_' + param for param in self.packet.fit_result.param_names)
        fit_params.extend('err_' + param for param in self.packet.fit_result.param_names)
        fit_params.extend(('chisq', 'ndof', 'mb', 'abs_mag'))
        columns = self.packet.fitted_params_to_pandas().columns
        np.testing.assert_array_equal(fit_params, columns)

    def test_values_match_params(self) -> None:
        """Test the values of the returned dataframe match simulated parameters"""

        returned_data = self.packet.fitted_params_to_pandas().iloc[0]

        self.assertEqual(self.packet.snid, returned_data['snid'], 'Incorrect SNID')
        self.assertEqual(self.packet.fit_result.chisq, returned_data['chisq'], 'Incorrect chisq')
        self.assertEqual(self.packet.fit_result.ndof, returned_data['ndof'], 'Incorrect ndof')

        for param in self.packet.fit_result.param_names:
            fit_result = self.packet.fitted_model[param]
            return_val = returned_data[f'fit_{param}']
            self.assertEqual(fit_result, return_val, f'Incorrect fit value for {param}')

            fit_err = self.packet.fit_result.errors.get(param, MASK_VALUE)
            return_err = returned_data[f'err_{param}']
            self.assertEqual(fit_err, return_err, f'Incorrect error value for {param}')

    def test_values_error_for_missing_fit_result(self) -> None:
        """Test returned values are masked when no fit results are stored in the packet"""

        # Create a packet with no fit results.  
        packet = create_mock_pipeline_packet(include_fit=False)
        with self.assertRaises(ValueError):
            packet.fitted_params_to_pandas()


class PacketStatusToPandas(TestCase):
    """Tests for the compilation of packet status indicators into a DataFrame"""

    def test_df_matches_packet_on_successful_fit(self):
        """Test the returned dataframe matches data from a picket with a successful fit result"""

        packet = create_mock_pipeline_packet()
        df_data = packet.packet_status_to_pandas().iloc[0]

        self.assertEqual(packet.snid, df_data['snid'])
        self.assertEqual(packet.message, df_data['message'])
        self.assertEqual(packet.fit_result.success, df_data['success'])

    def test_df_matches_packet_on_missing_fit(self):
        """Test the returned dataframe matches data from a picket with a failed fit result"""

        packet = create_mock_pipeline_packet(include_fit=False)
        df_data = packet.packet_status_to_pandas().iloc[0]

        self.assertEqual(packet.snid, df_data['snid'])
        self.assertEqual(packet.message, df_data['message'])
        self.assertEqual(False, df_data['success'])
