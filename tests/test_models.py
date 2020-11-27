"""Tests for the ``models`` module"""

import inspect
from copy import copy
from unittest import TestCase

import numpy as np
import pandas as pd
import sncosmo
from sncosmo.tests import test_models as sncosmo_test_models

from snat_sim import constants as const
from snat_sim import models
from snat_sim.filters import register_decam_filters
from tests.mock import create_constant_pwv_model

register_decam_filters(force=True)


class TestFixedResTransmission(TestCase):
    """Tests for the ``FixedResTransmission`` class"""

    def setUp(self):
        """Create a dummy ``Transmission`` object"""

        self.transmission = models.FixedResTransmission(4)

    def test_interpolation_on_grid_point(self):
        """Test interpolation result matches sampled values at the grid points"""

        test_pwv = self.transmission.samp_pwv[1]
        expected_transmission = self.transmission.samp_transmission[1]

        returned_trans = self.transmission(test_pwv)
        np.testing.assert_equal(expected_transmission, returned_trans)

    def test_default_wavelengths_match_sampled_wavelengths(self):
        """Test return values are index by sample wavelengths by default"""

        np.testing.assert_equal(self.transmission(4).index.to_numpy(), self.transmission.samp_wave)

    def test_interpolates_for_given_wavelengths(self):
        """Test an interpolation is performed for specified wavelengths when given"""

        test_pwv = self.transmission.samp_pwv[1]
        test_wave = np.arange(3000, 3500, 50)

        returned_wave = self.transmission(test_pwv, wave=test_wave).index.values
        np.testing.assert_equal(returned_wave, test_wave)

    def test_scalar_pwv_returns_series(self):
        """Test passing a scalar PWV value returns a pandas Series object"""

        transmission = self.transmission(4)
        self.assertIsInstance(transmission, pd.Series)
        self.assertEqual(transmission.name, f'4.0 mm')

    def test_vector_pwv_returns_dataframe(self):
        """Test passing a vector of PWV values returns a pandas DataFrame"""

        transmission = self.transmission([4, 5])
        self.assertIsInstance(transmission, pd.DataFrame)
        np.testing.assert_equal(transmission.columns.values, [f'4.0 mm', f'5.0 mm'])


class TestVariablePropagationEffect(TestCase):
    """Tests for the ``modeling.VariablePropagationEffect`` class"""

    def test_time_arg_in_signature(self):
        """Test the ``propagate`` method includes a time parameters"""

        params = list(inspect.signature(models.VariablePropagationEffect.propagate).parameters.keys())
        self.assertEqual(params[-1], 'time')


class TestStaticPWVTrans(TestCase):
    """Tests for the ``StaticPWVTrans`` class"""

    def setUp(self):
        self.transmission_effect = models.StaticPWVTrans()

    def test_default_pwv_is_zero(self):
        """Test the default ``pwv`` parameter is 0"""

        self.assertEqual(0, self.transmission_effect['pwv'])

    def test_propagation_applies_pwv_transmission(self):
        """Test the ``propagate`` applies PWV absorption"""

        # Get the expected transmission
        pwv = 5

        wave = np.arange(4000, 5000)
        transmission_model = models.FixedResTransmission(res=self.transmission_effect.transmission_res)
        transmission = transmission_model(pwv=pwv, wave=wave)

        # Get the expected flux
        flux = np.ones_like(wave)
        expected_flux = flux * transmission

        # Get the returned flux
        self.transmission_effect._parameters = [pwv]
        propagated_flux = self.transmission_effect.propagate(wave, flux)
        np.testing.assert_equal(expected_flux, propagated_flux[0])


class TestVariablePWVTrans(TestCase):
    """Tests for the ``modeling.VariablePWVTrans`` class"""

    def setUp(self):
        self.constant_pwv = 4
        self.mock_pwv_model = create_constant_pwv_model(self.constant_pwv)
        self.propagation_effect = models.VariablePWVTrans(self.mock_pwv_model)

    def test_parameter_arrays_match_length(self):
        """Test parameter array and parameter names have same length"""

        num_param_names = len(self.propagation_effect._param_names)
        self.assertEqual(
            len(self.propagation_effect._parameters), num_param_names,
            'Number of parameters does not match number of parameter names.')

        self.assertEqual(
            len(self.propagation_effect.param_names_latex), num_param_names,
            'Number of parameters does not match number of parameter LATEX names.')

    def test_default_location_params_match_vro(self):
        """Test the default values for the observer location match VRO"""

        self.assertEqual(self.propagation_effect['lat'], const.vro_latitude)
        self.assertEqual(self.propagation_effect['lon'], const.vro_longitude)
        self.assertEqual(self.propagation_effect['alt'], const.vro_altitude)

    def test_propagation_includes_pwv_transmission(self):
        """Test propagated flux includes absorption from PWV"""

        wave = np.arange(3000, 12000)
        flux = np.ones_like(wave)
        propagated_flux = self.propagation_effect.propagate(wave, flux, time=0)
        np.testing.assert_array_less(propagated_flux, flux)


class TestSNModel(sncosmo_test_models.TestModel, TestCase):
    """Tests for the ``modeling.SNModel`` class

    Includes all tests written for the ``sncosmo.Model`` class.
    """

    def setUp(self):
        self.model = models.SNModel(
            source=sncosmo_test_models.flatsource(),
            effects=[sncosmo.CCM89Dust()],
            effect_frames=['obs'],
            effect_names=['mw'])

        self.model.set(z=0.0001)

    def test_copy_returns_correct_type(self):
        """Test copied objects are of ``modeling.SNModel`` type"""

        copied = copy(self.model)
        self.assertIsInstance(copied, models.SNModel)

    def test_copy_copies_parameters(self):
        """Test parameter values are copied to new id values"""

        copied = copy(self.model)
        for original_param, copied_param in zip(self.model._parameters, copied.parameters):
            self.assertNotEqual(id(original_param), id(copied_param))
            self.assertEqual(original_param, copied_param)

    def test_copied_parameters_are_not_linked(self):
        """Test parameters of a copied model are independent from the original model"""

        old_params = copy(self.model.parameters)
        copied_model = copy(self.model)
        copied_model.set(z=1)
        np.testing.assert_equal(old_params, self.model.parameters)

    def test_error_for_bad_frame(self):
        """Test an error is raised for a band reference frame name"""

        model = models.SNModel(source='salt2')
        with self.assertRaises(ValueError):
            model.add_effect(effect=sncosmo.CCM89Dust(), frame='bad_frame_name', name='mw')

    def test_free_effect_adds_z_parameter(self):
        """Test effects in the ``free`` frame of reference include an added redshift parameter"""

        effect_name = 'freeMW'
        model = models.SNModel(source='salt2')
        model.add_effect(effect=sncosmo.CCM89Dust(), frame='free', name=effect_name)
        self.assertIn(effect_name + 'z', model.param_names)

    def test_variable_propagation_support(self):
        """Test a time variable effect can be added and called without error"""

        effect = models.VariablePWVTrans(create_constant_pwv_model())
        model = models.SNModel(sncosmo_test_models.flatsource())
        model.add_effect(effect=effect, frame='obs', name='Variable PWV')
        model.flux(time=0, wave=[4000])

    def test_sed_matches_sncosmo_model(self):
        """Test the SED returned by the ``modeling.SNModel`` class matches the ``sncosmo.SNModel`` class"""

        wave = np.arange(3000, 12000)
        sncosmo_model = sncosmo.Model('salt2-extended')
        sncosmo_flux = sncosmo_model.flux(0, wave)
        custom_model = models.SNModel(sncosmo_model.source)
        custom_flux = custom_model.flux(0, wave)
        np.testing.assert_equal(custom_flux, sncosmo_flux)
