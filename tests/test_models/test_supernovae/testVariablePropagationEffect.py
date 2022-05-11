"""Tests for the ``snat_sim.modeling.supernova.VariablePropagationEffect`` class"""

import inspect
from unittest import TestCase

from snat_sim.models.supernova import VariablePropagationEffect


class PropagateMethodSignature(TestCase):
    """Tests for the signature of the ``propagate`` method

    A correct signature is required to maintain compatibility with ``sncosmo``.
    """

    def test_time_arg_in_signature(self) -> None:
        """Test the ``propagate`` method includes ``time`` as the LAST parameter

        This is important for maintaining reverse compatibility with the sncosmo package.
        """

        params = list(inspect.signature(VariablePropagationEffect.propagate).parameters.keys())
        self.assertEqual(params[-1], 'time')
