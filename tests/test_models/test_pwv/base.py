"""A collection of reusable base tests that are generally applicable to
multiple classes in the `snat_sim.modeling.pwv`` module.
"""

import numpy as np


class PropagationEffectTests:
    """Base tests for the construction of propagation effects"""

    def test_param_names_match_number_parameters(self):
        """Test parameter array and parameter names have same length"""

        self.assertEqual(
            len(self.propagation_effect._parameters),
            len(self.propagation_effect._param_names),
            'Number of parameters does not match number of parameter names.')

    def test_latex_names_matches_number_parameters(self):
        """Test parameter latex descriptions and parameter names have same length"""

        self.assertEqual(
            len(self.propagation_effect._parameters),
            len(self.propagation_effect.param_names_latex),
            'Number of parameters does not match number of parameter LATEX names.')

    def test_propagation_includes_pwv_transmission(self):
        """Test propagated flux includes absorption from PWV"""

        wave = np.arange(3000, 12000)
        flux = np.ones_like(wave)
        propagated_flux = self.propagation_effect.propagate(wave, flux, time=0)
        np.testing.assert_array_less(propagated_flux, flux)
