"""The ``models`` module defines classes for modeling different
physical phenomena. This includes SNe Ia light-curves, the propagation of
light through atmospheric water vapor (with and without variation in time),
and the seasonal variation of precipitable water vapor over time.

Model Summaries
---------------

A summary of the available models is provided below:

.. autosummary::
   :nosignatures:

   FixedResTransmission
   PWVModel
   SNModel

Supernova models (``SNModel``) are designed to closely resemble the behavior
of the ``sncosmo`` package. However, unlike ``sncosmo.Model`` objects, the
``snat_sim.SNModel`` class provides support for propogation effects that vary
with time. A summary of propagation effects provided by the ``snat_sim``
package is listed below:

.. autosummary::
   :nosignatures:

   StaticPWVTrans
   SeasonalPWVTrans
   VariablePWVTrans

Usage Example
-------------

To ensure backwards compatibility and ease of use, supernovae modeling with the
``snat_sim`` package follows (but also extends) the same
`design pattern <https://sncosmo.readthedocs.io/en/stable/models.html>`_
as the ``sncosmo`` package. Models are instantiated for a given spectral
template and various propagation effects can be added to the model. In the
following example, atmospheric propagation effects due to precipitable water
vapor are added to a Salt2 supernova model.

.. doctest:: python

   >>> from snat_sim import modeling

   >>> # Create a supernova model
   >>> supernova_model = modeling.SNModel('salt2')

   >>> # Create a model for the atmosphere
   >>> atm_transmission = modeling.StaticPWVTrans()
   >>> atm_transmission.set(pwv=4)
   >>> supernova_model.add_effect(effect=atm_transmission, name='Atmosphere', frame='obs')


To simulate a light-curve, you must first establish the desired light-curve
cadence (i.e., how the light-curve should be sampled in time):

.. doctest:: python

   >>> cadence = modeling.ObservedCadence(
   ...     obs_times=[-1, 0, 1],
   ...     bands=['sdssr', 'sdssr', 'sdssr'],
   ...     zp=25, zpsys='AB', skynoise=0, gain=1
   ... )

Light-curves can then be simulated directly from the model:

.. doctest:: python

   >>> # Here we simulate a light-curve with statistical noise
   >>> light_curve = supernova_model.simulate_lc(cadence)

   >>> # Here we simulate a light-curve with a fixed signal to noise ratio
   >>> light_curve_fixed_snr = supernova_model.simulate_lc(cadence, fixed_snr=5)


Module Docs
-----------
"""

from .pwv import *
from .reference_star import *
from .supernova import *
