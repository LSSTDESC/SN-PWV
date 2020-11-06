"""Tests for the ``models`` module"""

import inspect
from copy import copy
from unittest import TestCase

import numpy as np
import sncosmo
from pwv_kpno.defaults import v1_transmission
from sncosmo.tests import test_models as sncosmo_test_models

from snat_sim import constants as const
from snat_sim import models
from snat_sim.filters import register_decam_filters

register_decam_filters(force=True)


class TestVariablePropagationEffect(TestCase):
    """Tests for the ``modeling.VariablePropagationEffect`` class"""

    def test_time_arg_in_signature(self):
        """Test the ``propagate`` method includes a time parameters"""

        params = list(inspect.signature(models.VariablePropagationEffect.propagate).parameters.keys())
        self.assertEqual(params[-1], 'time')


class TestVariablePWVTrans(TestCase):
    """Tests for the ``modeling.VariablePWVTrans`` class"""

    def setUp(self):
        self.default_pwv = 5
        self.constant_pwv_func = lambda date, format: self.default_pwv
        self.propagation_effect = models.VariablePWVTrans(self.constant_pwv_func)

    def test_default_location_params_match_vro(self):
        """Test the default values for the observer location match VRO"""

        self.assertEqual(self.propagation_effect['lat'], const.vro_latitude)
        self.assertEqual(self.propagation_effect['lon'], const.vro_longitude)
        self.assertEqual(self.propagation_effect['alt'], const.vro_altitude)

    def test_airmass_scaling_on_by_default(self):
        """Test airmass scaling is turned on by default"""

        self.assertTrue(self.propagation_effect.scale_airmass)

    def test_transmission_version_support(self):
        """Test the propagation object uses the atmospheric model corresponding specified at init"""

        from pwv_kpno.transmission import CrossSectionTransmission

        default_effect = models.VariablePWVTrans(self.constant_pwv_func)
        self.assertIsInstance(default_effect._transmission_model, CrossSectionTransmission)

        v1_effect = models.VariablePWVTrans(self.constant_pwv_func, transmission_version='v1')
        self.assertIsInstance(v1_effect._transmission_model, CrossSectionTransmission)

        # Todo: When the v2 model is available, add a test condition
        # v2_effect = modeling.VariablePWVTrans(self.constant_pwv_func, transmission_version='v2')
        # self.assertIsInstance(v2_effect._transmission_model, TransmissionModel)

        with self.assertRaises(ValueError):
            models.VariablePWVTrans(self.constant_pwv_func, transmission_version='NotAVersion')

    def test_propagation_includes_pwv_transmission(self):
        """Test propagated flux includes absorption from PWV"""

        wave = np.arange(3000, 12000)
        flux = np.ones_like(wave)

        self.propagation_effect.scale_airmass = False
        transmission = self.propagation_effect._transmission_model(
            self.default_pwv, wave, self.propagation_effect['res'])

        propagated_flux = self.propagation_effect.propagate(wave, flux, time=0)
        np.testing.assert_equal(propagated_flux, transmission.values)

    def test_pwv_los_is_scaled_by_airmass(self):
        """Test PWV is scaled by airmass when the ``scale_airmass`` attribute is ``True``"""

        self.propagation_effect.scale_airmass = True
        self.propagation_effect.set(ra=2, dec=2)
        airmass = self.propagation_effect.airmass(time=0)
        pwv_los = self.propagation_effect.calc_pwv_los(time=0)
        self.assertEqual(pwv_los, self.default_pwv * airmass)


class TestModel(sncosmo_test_models.TestModel, TestCase):
    """Tests for the ``modeling.Model`` class

    Includes all tests written for the ``sncosmo.Model`` class.
    """

    def setUp(self):
        self.model = models.Model(
            source=sncosmo_test_models.flatsource(),
            effects=[sncosmo.CCM89Dust()],
            effect_frames=['obs'],
            effect_names=['mw'])

        self.model.set(z=0.0001)

    def test_copy_returns_correct_type(self):
        """Test copied objects are of ``modeling.Model`` type"""

        copied = copy(self.model)
        self.assertIsInstance(copied, models.Model)

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

        model = models.Model(source='salt2')
        with self.assertRaises(ValueError):
            model.add_effect(effect=sncosmo.CCM89Dust(), frame='bad_frame_name', name='mw')

    def test_free_effect_adds_z_parameter(self):
        """Test effects in the ``free`` frame of reference include an added redshift parameter"""

        effect_name = 'freeMW'
        model = models.Model(source='salt2')
        model.add_effect(effect=sncosmo.CCM89Dust(), frame='free', name=effect_name)
        self.assertIn(effect_name + 'z', model.param_names)

    def test_variable_propagation_support(self):
        """Test a time variable effect can be added and called without error"""

        effect = models.VariablePWVTrans(lambda date, format: 5)
        model = models.Model(sncosmo_test_models.flatsource())
        model.add_effect(effect=effect, frame='obs', name='Variable PWV')
        model.flux(time=0, wave=[4000])

    def test_sed_matches_sncosmo_model(self):
        """Test the SED returned by the ``modeling.Model`` class matches the ``sncosmo.Model`` class"""

        wave = np.arange(3000, 12000)
        sncosmo_model = sncosmo.Model('salt2-extended')
        sncosmo_flux = sncosmo_model.flux(0, wave)
        custom_model = models.Model(sncosmo_model.source)
        custom_flux = custom_model.flux(0, wave)
        np.testing.assert_equal(custom_flux, sncosmo_flux)


class TestPWVTrans(TestCase):
    """Tests for the addition of PWV to sncosmo models"""

    def setUp(self):
        self.transmission_effect = models.StaticPWVTrans()

    def test_default_pwv_is_zero(self):
        """Test the default ``pwv`` parameter is 0"""

        self.assertEqual(0, self.transmission_effect['pwv'])

    def test_default_resolution_is_five(self):
        """Test the default ``res`` parameter is 5"""

        self.assertEqual(5, self.transmission_effect['res'])

    def test_propagation_applies_pwv_transmission(self):
        """Test the ``propagate`` applies PWV absorption"""

        # Get the expected transmission
        pwv = res = 5
        wave = np.arange(4000, 5000)
        transmission = v1_transmission(pwv=pwv, wave=wave, res=res)

        # Get the expected flux
        flux = np.ones_like(wave)
        expected_flux = flux * transmission

        # Get the returned flux
        self.transmission_effect._parameters = [pwv, res]
        propagated_flux = self.transmission_effect.propagate(wave, flux)
        np.testing.assert_equal(expected_flux, propagated_flux[0])