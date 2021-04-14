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

Usage Example
-------------

To ensure ease of use, supernova modeling with the ``snat_sim`` package is
backward compatible with the ``sncosmo`` package and follows a similar
`design pattern <https://sncosmo.readthedocs.io/en/stable/models.html>`_.
The ``SNModel`` class from ``snat_sim`` acts as a drop-in replacement for
the ``Model`` classes built into ``sncosmo`` but provides extended
functionality.

Like ``sncosmo``, models are instantiated for a given spectral
template in addition to observer or rest-frame propagation effects. In the
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

   >>> from snat_sim.models import LightCurve

   >>> light_curve_data = LightCurve(
   ... time=[55070.000000, 55072.051282, 55074.102564, 55076.153846],
   ... band=['sdssg', 'sdssr', 'sdssi', 'sdssz'],
   ... flux=[0.363512, -0.200801,  0.307494,  1.087761],
   ... fluxerr=[0.672844, 0.672844, 0.672844, 0.672844],
   ... zp=[25.0, 25.0, 25.0, 25.0],
   ... zpsys=['ab', 'ab', 'ab', 'ab'])

   >>> light_curve_data.to_pandas()
                  band      flux   fluxerr    zp zpsys  phot_flag
   time
   55070.000000  sdssg  0.363512  0.672844  25.0    ab        0.0
   55072.051282  sdssr -0.200801  0.672844  25.0    ab        0.0
   55074.102564  sdssi  0.307494  0.672844  25.0    ab        0.0
   55076.153846  sdssz  1.087761  0.672844  25.0    ab        0.0


   >>> light_curve_data.to_astropy()
       time      band    flux   fluxerr     zp   zpsys phot_flag
     float64     str5  float64  float64  float64  str2  float64
   ------------ ----- --------- -------- ------- ----- ---------
        55070.0 sdssg  0.363512 0.672844    25.0    ab       0.0
   55072.051282 sdssr -0.200801 0.672844    25.0    ab       0.0
   55074.102564 sdssi  0.307494 0.672844    25.0    ab       0.0
   55076.153846 sdssz  1.087761 0.672844    25.0    ab       0.0


Fitting Light-Curves
^^^^^^^^^^^^^^^^^^^^

Photometric data can be fit directly from the model using the ``fit_lc`` method.
Notice in the below example that the returned object types are classes from the
``snat_sim`` package:

.. doctest:: python

   >>> fit_result, fitted_model = supernova_model.fit_lc(light_curve, vparam_names=['x0', 'x1', 'c'])
   >>> print(type(fit_result), type(fitted_model))
   <class 'snat_sim.models.supernova.SNFitResult'> <class 'snat_sim.models.supernova.SNModel'>

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
distance modulus given the alpha/beta standardization parameters:

.. doctest:: python

   >>> from snat_sim import constants as const

   >>> fit_result.salt_covariance_linear()

   >>> # Here we use alpha and beta parameters from Betoule et al. 2014
   >>> mu_variance = fit_result.mu_variance_linear(alpha=const.betoule_alpha, beta=const.betoule_beta)
   >>> print(f'{mu_variance: .5f}')
    0.00735

Calibrating Light-Curves
^^^^^^^^^^^^^^^^^^^^^^^^

Your use case may involve calibrating simulated light-curves relative to a
stellar reference catalog. The spectrum for individual spectral types can be retreived
using the ``ReferenceStar`` class:

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

   >>> import sncosmo

   >>> light_curve = sncosmo.load_example_data()
   >>> reference_catalog = reference_star.ReferenceCatalog('G2', 'M5')
   >>> print(reference_catalog.calibrate_lc(light_curve, pwv=4))

Module Docs
-----------
"""

from .light_curve import *
from .pwv import *
from .reference_star import *
from .supernova import *
