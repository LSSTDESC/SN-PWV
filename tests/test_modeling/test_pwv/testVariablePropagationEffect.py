"""Tests for the ``VariablePropagationEffect`` class"""

import inspect
from unittest import TestCase

from snat_sim.modeling import pwv


class PropagateMethodSignature(TestCase):
    """Tests for the signature of the ``propagate`` method

    A correct signature is required to maintain compatibility with ``sncosmo``.
    """

    def test_time_arg_in_signature(self):
        """Test the ``propagate`` method includes a ``time`` as the last parameter"""

        params = list(inspect.signature(pwv.VariablePropagationEffect.propagate).parameters.keys())
        self.assertEqual(params[-1], 'time')
