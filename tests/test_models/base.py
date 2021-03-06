"""A collection of test cases that are generally applicable to multiple
object types.
"""

import numpy as np


class PropagationEffectTests:
    """Base tests for the construction of propagation effects"""

    def test_parameters_match_number_param_names(self):
        """Test parameter array and parameter names have same length"""

        self.assertEqual(
            len(self.propagation_effect._parameters),
            len(self.propagation_effect._param_names),
            'Number of parameters does not match number of parameter names.')

    def test_latex_matches_number_param_names(self):
        """Test parameter latex descriptions and parameter names have same length"""

        self.assertEqual(
            len(self.propagation_effect.param_names_latex),
            len(self.propagation_effect._param_names),
            'Number of parameters does not match number of parameter LATEX names.')

    def test_propagation_includes_pwv_transmission(self):
        """Test propagated flux includes absorption from PWV"""

        wave = np.arange(3000, 12000)
        flux = np.ones_like(wave)
        propagated_flux = self.propagation_effect.propagate(wave, flux, time=0)
        np.testing.assert_array_less(propagated_flux, flux)
