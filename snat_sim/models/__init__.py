"""The ``modeling`` module defines classes for modeling physical phenomena.
This includes SNe Ia light-curves, the propagation of light through atmospheric
water vapor (with and without variation in time), and the broad-band photometry
of stellar catalogs.

Model Summaries
---------------

A summary of the available models is provided below:

.. autosummary::
   :nosignatures:

   PWVTransmissionModel
   PWVModel
   SNModel
   ReferenceStar
   ReferenceCatalog
   VariableCatalog

Supernova models (``SNModel``) are designed to closely resemble the behavior
of the ``sncosmo`` package. However, unlike ``sncosmo.Model`` objects, the
``snat_sim.SNModel`` class provides support for propagation effects that vary
with time. A summary of custom propagation effects provided by the ``snat_sim``
package is listed below:

.. autosummary::
   :nosignatures:

   StaticPWVTrans
   SeasonalPWVTrans
   VariablePWVTrans

Simulation results are generally returned using one or more of the following
object types:

.. autosummary::
   :nosignatures:

   ObservedCadence
   LightCurve
   SNFitResult

Usage Example
-------------

To ensure ease of use, supernova modeling with the ``snat_sim`` package is
backward compatible with the ``sncosmo`` package and follows a similar
`design pattern <https://sncosmo.readthedocs.io/en/stable/models.html>`_.
The ``SNModel`` class from ``snat_sim`` acts as a drop-in replacement for
the ``Model`` classes built into ``sncosmo`` but provides extended
functionality.

Supernova models are instantiated using a spectral template with the addition of
optional observer or rest-frame effects. In the
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
handled directly from the model class. We consider these two cases seperately
in the following sections.

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
``simulate_lc`` method. Notice that the returned type is a ``LightCurve``
object:

.. doctest:: python

   >>> # Here we simulate a light-curve with statistical noise
   >>> light_curve_with_noise = supernova_model.simulate_lc(cadence)
   >>> print(type(light_curve_with_noise))
   <class 'snat_sim.models.light_curve.LightCurve'>

   >>> # Here we simulate a light-curve with a fixed signal to noise ratio
   >>> light_curve_fixed_snr = supernova_model.simulate_lc(cadence, fixed_snr=5)
   >>> print(type(light_curve_fixed_snr))
   <class 'snat_sim.models.light_curve.LightCurve'>

The ``LightCurve`` class represents astronomical light-curves and provides an
easy interface for casting the data into other commonly used object types.

.. doctest:: python

   >>> import sncosmo
   >>> from snat_sim.models import LightCurve

   >>> example_data = sncosmo.load_example_data()
   >>> light_curve_data = LightCurve.from_sncosmo(example_data)

   >>> lc_as_dataframe = light_curve_data.to_pandas()
   >>> lc_as_table = light_curve_data.to_astropy()

Fitting Light-Curves
^^^^^^^^^^^^^^^^^^^^

Photometric data can be fit directly from the model using the ``fit_lc`` method.
Notice in the below example that the returned object types are classes from the
``snat_sim`` package:

.. doctest:: python

   >>> supernova_model.set(z=.5, t0=55100.0)
   >>> fit_result, fitted_model = supernova_model.fit_lc(
   ...     light_curve_data, vparam_names=['x0', 'x1', 'c'])

   >>> print(type(fit_result))
   <class 'snat_sim.models.supernova.SNFitResult'>

   >>> print(type(fitted_model))
   <class 'snat_sim.models.supernova.SNModel'>

The ``SNFitResult`` object is similar to the ``Result`` class from ``sncosmo``
but **is not backward compatible**. ``SNFitResult`` instances provide access
to fit results as floats or ``pandas`` objects (e.g., ``pandas.Series`` or
``pandas.DataFrame``) depending on the value.

.. doctest:: python

   >>> print(fit_result)
        success: True
        message: Minimization exited successfully.
          ncall: 77
           nfit: 1
          chisq: 35.537
           ndof: 37
    param_names: ['z', 't0', 'x0', 'x1', 'c', 'Atmospherepwv']
     parameters: [5.000e-01 5.510e+04 1.194e-05 4.257e-01 2.483e-01 4.000e+00]
   vparam_names: ['x0', 'x1', 'c']
         errors: [3.830e-07 3.173e-01 2.931e-02]
     covariance:
       [[ 1.467e-13 -6.472e-08 -7.966e-09]
        [-6.472e-08  1.007e-01  1.174e-03]
        [-7.966e-09  1.174e-03  8.590e-04]]

Fit result objects are also capable of calculating the variance in the
distance modulus given the alpha/beta standardization parameters:

.. doctest:: python

   >>> from snat_sim import constants as const

   >>> # The covariance matrix used when determining error values
   >>> fit_result.salt_covariance_linear()
             mB        x1         c
   mB  0.001213  0.005886  0.000725
   x1  0.005886  0.100684  0.001174
   c   0.000725  0.001174  0.000859

   >>> # Here we use alpha and beta parameters from Betoule et al. 2014
   >>> mu_variance = fit_result.mu_variance_linear(
   ...     alpha=const.betoule_alpha, beta=const.betoule_beta)

   >>> print(f'{mu_variance: .5f}')
    0.00762

Calibrating Light-Curves
^^^^^^^^^^^^^^^^^^^^^^^^

Your use case may involve calibrating simulated light-curves relative to a
stellar reference catalog. The spectrum for individual spectral types can be
retreived using the ``ReferenceStar`` class:

.. doctest:: python

   >>> from snat_sim.models import reference_star

   >>> g2_star = reference_star.ReferenceStar('G2')
   >>> print(g2_star.to_pandas())
   3000.000     4.960049e+17
   3000.006     4.659192e+17
   3000.012     4.304657e+17
   3000.018     3.751426e+17
   3000.024     2.847191e+17
                    ...
   11999.920    1.366567e+18
   11999.940    1.366673e+18
   11999.960    1.366418e+18
   11999.980    1.365863e+18
   12000.000    1.365315e+18
   Length: 933333, dtype: float32


A ``ReferenceCatalog`` is used to represent a collection of stars with different
stellar types. Catalog instances can be used to calibrate supernoca light-curves.

.. code-block:: python

   >>> reference_catalog = reference_star.ReferenceCatalog('G2', 'M5')
   >>> print(reference_catalog.calibrate_lc(light_curve_data, pwv=4))

Module Docs
-----------
"""

from .light_curve import *
from .pwv import *
from .reference_star import *
from .supernova import *
