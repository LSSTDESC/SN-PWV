"""Tests for the ``SNModel`` class"""

from copy import copy
from unittest import TestCase, skipIf

import numpy as np
import sncosmo
from sncosmo.tests import test_models as sncosmo_test_models

from snat_sim import models
from tests.mock import create_constant_pwv_model

try:
    import emcee

    no_emcee_package = False

except:
    no_emcee_package = True


class SncosmoBaseTests(sncosmo_test_models.TestModel, TestCase):
    """Includes all tests written for the ``sncosmo.Model`` class."""

    def setUp(self):
        # Same as the base sncosmo setup procedure, but using a ``SNModel``
        # instance instead of ``sncosmo.Model``
        self.model = models.SNModel(
            source=sncosmo_test_models.flatsource(),
            effects=[sncosmo.CCM89Dust()],
            effect_frames=['obs'],
            effect_names=['mw'])

        self.model.set(z=0.0001)


class BackwardsCompatibility(TestCase):
    """Test backwards compatibility with ``sncosmo.Model`` objects"""

    @staticmethod
    def test_sed_matches_sncosmo_model():
        """Test the SED returned by the ``modeling.SNModel`` class matches the ``sncosmo.Model`` class"""

        wave = np.arange(3000, 12000)
        sncosmo_model = sncosmo.Model('salt2-extended')
        sncosmo_flux = sncosmo_model.flux(0, wave)
        custom_model = models.SNModel(sncosmo_model.source)
        custom_flux = custom_model.flux(0, wave)
        np.testing.assert_equal(custom_flux, sncosmo_flux)

    def test_fit_lc_returns_correct_type(self):
        """Test the fit_lc function returns an ``SNModel`` instance for the fitted model"""

        data = sncosmo.load_example_data()
        model = models.SNModel('salt2')
        _, fitted_model = sncosmo.fit_lc(data, model, ['x0'])
        self.assertIsInstance(fitted_model, models.SNModel)

    @skipIf(no_emcee_package, 'emcee package is not installed')
    def test_mcmc_lc_returns_correct_type(self):
        """Test the fit_lc function returns an ``SNModel`` instance for the fitted model"""

        data = sncosmo.load_example_data()
        model = models.SNModel('salt2')
        _, fitted_model = sncosmo.mcmc_lc(data, model, ['x0'])
        self.assertIsInstance(fitted_model, models.SNModel)


class Copy(TestCase):
    """Test copying a model correctly copies the underlying class data"""

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


class PropagationSupport(TestCase):
    """Tests for the handling of propagation effects"""

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
