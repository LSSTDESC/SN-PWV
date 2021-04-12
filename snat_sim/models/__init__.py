"""The ``modeling`` module defines classes for modeling different
physical phenomena. This includes SNe Ia light-curves, the propagation of
light through atmospheric water vapor (with and without variation in time),
and the seasonal variation of precipitable water vapor over time.

Model Summaries
---------------

A summary of the available models is provided below:

.. autosummary::
   :nosignatures:

   PWVTransmissionModel
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

To ensure ease of use, supernovae modeling with the ``snat_sim`` package is
backwards compatibile with the ``sncosmo`` package and follows a similar
`design pattern <https://sncosmo.readthedocs.io/en/stable/models.html>`_.
The ``SNModel`` class from ``snat_sim`` acts as a drop in replacements for
the ``Model`` classes built in to ``sncosmo`` but provides extended
functionality.

Like ``sncosmo``, models are instantiated for a given spectral
template and various propagation effects can be added to the model. In the
following example, atmospheric propagation effects due to precipitable water
vapor are added to a Salt2 supernova model.

.. doctest:: python

   >>> from snat_sim import models

   >>> # Create a supernova model
   >>> supernova_model = models.SNModel('salt2')

   >>> # Create a model for the atmosphere
   >>> atm_transmission = models.StaticPWVTrans()
   >>> atm_transmission.set(pwv=4)
   >>> supernova_model.add_effect(effect=atm_transmission, name='Atmosphere', frame='obs')

Unlike ``sncosmo``, the process of simulating or fitting light-curves is
handled directly from the model class. Lets consider these two cases seperately.

Simulating Light-Curves
^^^^^^^^^^^^^^^^^^^^^^^

To simulate a light-curve, you first need to establish the desired light-curve
cadence (i.e., how the light-curve should be sampled in time):

.. doctest:: python

   >>> cadence = models.ObservedCadence(
   ...     obs_times=[-1, 0, 1],
   ...     bands=['sdssr', 'sdssr', 'sdssr'],
   ...     zp=25, zpsys='AB', skynoise=0, gain=1
   ... )

Light-curves can then be simulated directly from the model using the
``simulate_lc`` method:

.. doctest:: python

   >>> # Here we simulate a light-curve with statistical noise
   >>> light_curve = supernova_model.simulate_lc(cadence)

   >>> # Here we simulate a light-curve with a fixed signal to noise ratio
   >>> light_curve_fixed_snr = supernova_model.simulate_lc(cadence, fixed_snr=5)

Fitting Light-Curves
^^^^^^^^^^^^^^^^^^^^

Photometric data can be fit directly from the model using the ``fit_lc`` method.
Notice in the below example that the returned object types are classes from the
``snat_sim`` package:

.. doctest:: python

   >>> import sncosmo
   >>> from snat_sim.models import SNModel

   >>> data = sncosmo.load_example_data()
   >>> model_to_fit = SNModel('salt2')

   >>> # We set fix the z and t0 and then vary the remaining parameters during the fit
   >>> model_to_fit.set(z=data.meta['z'], t0=data.meta['t0'])
   >>> fit_result, fitted_model = model_to_fit.fit_lc(data, vparam_names=['x0', 'x1', 'c'])

   >>> print(type(fit_result), type(fitted_model))
   <class 'snat_sim.models.supernova.SNFitResult'> <class 'snat_sim.models.supernova.SNModel'>

The ``SNFitResult`` object is similar to the ``Result`` class from ``sncosmo``
but **is not backwards compatible**. ``SNFitResult`` instances provide access
to fit results as floats or ``pandas`` objects (e.g., ``pandas.Series`` or
``pandas.DataFrame``) depending on the value.

.. doctest:: python

   >>> print(fit_result)
        success: True
        message: Minimization exited successfully.
          ncall: 77
           nfit: 1
          chisq: 35.863
           ndof: 37
    param_names: ['z', 't0', 'x0', 'x1', 'c']
     parameters: [5.000e-01 5.510e+04 1.188e-05 4.382e-01 2.190e-01]
   vparam_names: ['x0', 'x1', 'c']
         errors: [3.777e-07 3.178e-01 2.862e-02]
     covariance:
       [[ 1.426e-13 -6.461e-08 -7.622e-09]
        [-6.461e-08  1.010e-01  1.223e-03]
        [-7.622e-09  1.223e-03  8.189e-04]]

Fit result objects are also capable of calculating the variance in the
distance modulus given the alpha/beta standerdization parameters:

.. doctest:: python

   >>> from snat_sim import constants as const

   >>> alpha = const.betoule_alpha
   >>> beta = const.betoule_beta
   >>> mu_variance = fit_result.mu_variance_linear(alpha, beta)
   >>> print(f'{mu_variance: .5f}')
    0.00735

Module Docs
-----------
"""

from .light_curve import *
from .pwv import *
from .reference_star import *
from .supernova import *
