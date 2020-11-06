"""The ``constants`` module defines values for constants used across the
parent package.

Value Summaries
---------------

+------------------------------+----------------------------------------------+
| Value                        | Description                                  |
+==============================+==============================================+
| ``vro_latitude``             | Latitude of the Vera Rubin Observatory in    |
|                              | degrees.                                     |
+------------------------------+----------------------------------------------+
| ``vro_longitude``            | Longitude of the Vera Rubin Observatory in   |
|                              | degrees.                                     |
+------------------------------+----------------------------------------------+
| ``vro_altitude``             | Altitude of the Vera Rubin Observatory in    |
|                              | meters.                                      |
+------------------------------+----------------------------------------------+
| ``betoule_alpha``            | Nuisance parameter alpha                     |
|                              | from Betoule+ 2014.                          |
+------------------------------+----------------------------------------------+
| ``betoule_beta``             | Nuisance parameter beta                      |
|                              | from Betoule+ 2014.                          |
+------------------------------+----------------------------------------------+
| ``betoule_omega_m``          | Cosmological mater density as determined in  |
|                              | Betoule+ 2014.                               |
+------------------------------+----------------------------------------------+
| ``betoule_abs_mb``           | Intrinsic absolute magnitude of SNe Ia as    |
|                              | determined in Betoule+ 2014.                 |
+------------------------------+----------------------------------------------+
| ``betoule_H0``               | Cosmological Hubble parameter as             |
|                              | determined in Betoule+ 2014.                 |
+------------------------------+----------------------------------------------+
| ``betoule_cosmo``            | ``astropy.Cosmology`` object representing    |
|                              | the best fit cosmology in Betoule+ 2014.     |
+------------------------------+----------------------------------------------+
"""

from astropy.cosmology import FlatLambdaCDM as _FlatLambdaCDM

# Vera Rubin Observatory coordinates from Google Maps
vro_latitude = -30.244573
vro_longitude = -70.7499537
vro_altitude = 1024

# Fitted parameters from the cosmological analysis of Betoule et al. 2014
betoule_alpha = 0.141
betoule_beta = 3.101
betoule_omega_m = 0.295
betoule_abs_mb = -19.05
betoule_H0 = 70
betoule_cosmo = _FlatLambdaCDM(H0=betoule_H0, Om0=betoule_omega_m)
