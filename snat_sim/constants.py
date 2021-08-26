"""The ``constants`` module defines values for physical constants used across
the parent package. Units used for each value are summarized in the table
below.

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
|                              | from Betoule+ 2014 (units of magnitudes).    |
+------------------------------+----------------------------------------------+
| ``betoule_beta``             | Nuisance parameter beta                      |
|                              | from Betoule+ 2014 (units of magnitudes).    |
+------------------------------+----------------------------------------------+
| ``betoule_omega_m``          | Cosmological mater density as determined in  |
|                              | Betoule+ 2014 (Dimensionless).               |
+------------------------------+----------------------------------------------+
| ``betoule_abs_mb``           | Intrinsic absolute magnitude of SNe Ia       |
|                              | from Betoule+ 2014 (units of magnitudes).    |
+------------------------------+----------------------------------------------+
| ``betoule_H0``               | Cosmological Hubble parameter from           |
|                              | Betoule+ 2014 in  km / s / Mpc.              |
+------------------------------+----------------------------------------------+
| ``betoule_cosmo``            | ``astropy.Cosmology`` object representing    |
|                              | the best fit cosmology in Betoule+ 2014.     |
+------------------------------+----------------------------------------------+
| ``jun_solstice``             | Date of the first solstice in 2020.          |
+------------------------------+----------------------------------------------+
| ``dec_solstice``             | Date of the second solstice in 2020.         |
+------------------------------+----------------------------------------------+
| ``mar_equinox``              | Date of the first equinox in 2020.           |
+------------------------------+----------------------------------------------+
| ``sep_equinox``              | Date of the second equinox in 2020.          |
+------------------------------+----------------------------------------------+
"""

from datetime import datetime

from astropy.cosmology import FlatLambdaCDM as _FlatLambdaCDM
from pytz import utc

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

# Start dates for each season
# 2020 was specifically chosen because it is a leap year.
# This makes certain datetime calculations easier
jun_solstice = datetime(2020, 6, 21, tzinfo=utc)
dec_solstice = datetime(2020, 12, 21, tzinfo=utc)
mar_equinox = datetime(2020, 3, 20, tzinfo=utc)
sep_equinox = datetime(2020, 9, 22, tzinfo=utc)
