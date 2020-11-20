:orphan:

PLaSTICC Data Model
===================

This page outlines the data model for PLaSTICC simulations used by the parent
project. It is **not a complete, formal, or official record** for
the described data model. It is intended only as a quick reference for use by developers.

Organizational Model
--------------------

Simulated light-curves are divided into directories based on the model used
in the simulation. For certain types of astronomical objects, multiple models
were used. Light-curves from different directories/models should not be taken
to represent distinct types or subtypes of objects. A summary of the models
is as follows:

+---------------------------+-----------------+
|  Model Number             |  Model Name     |
+===========================+=================+
|                  11       |    SNIa-normal  |
+---------------------------+-----------------+
|                   2       |        SNCC-II  |
+---------------------------+-----------------+
|                   3       |       SNCC-Ibc  |
+---------------------------+-----------------+
|                   2       |        SNCC-II  |
+---------------------------+-----------------+
|                   3       |       SNCC-Ibc  |
+---------------------------+-----------------+
|                   2       |        SNCC-II  |
+---------------------------+-----------------+
|                  41       |      SNIa-91bg  |
+---------------------------+-----------------+
|                  43       |         SNIa-x  |
+---------------------------+-----------------+
|                  51       |             KN  |
+---------------------------+-----------------+
|                  60       |         SLSN-I  |
+---------------------------+-----------------+
|                  99       |           PISN  |
+---------------------------+-----------------+
|                  99       |           ILOT  |
+---------------------------+-----------------+
|                  99       |           CART  |
+---------------------------+-----------------+
|                  64       |            TDE  |
+---------------------------+-----------------+
|                  70       |            AGN  |
+---------------------------+-----------------+
|                  80       |        RRlyrae  |
+---------------------------+-----------------+
|                  81       |         Mdwarf  |
+---------------------------+-----------------+
|                  83       |            EBE  |
+---------------------------+-----------------+
|                  84       |           MIRA  |
+---------------------------+-----------------+
|                  99       |   uLens-Binary  |
+---------------------------+-----------------+
|                  91       |    uLens-Point  |
+---------------------------+-----------------+
|                  99       |   uLens-STRING  |
+---------------------------+-----------------+
|                  91       |    uLens-Point  |
+---------------------------+-----------------+

File Format
-----------

Light-curves are saved using the `SNANA` file format where files come in
pairs: a header file postfixed with `HEAD.fits` and a photometry file
postfixed with `PHOT.fits`. The header file provides meta-data about the
observed targets (e.g., `RA` and `Dec`). The photometry file contains the
simulated light-curve. Each file containing information for multiple
supernovae. Definitions are provided below for a handful of columns in each
file type:

+--------------------+-------------------------------------------------------+
| Header File Column | Value Description                                     |
+====================+=======================================================+
| `SNID`             | Unique object identifier                              |
+--------------------+-------------------------------------------------------+
| `RA`, `DECL`       | On sky coordinates of the simulated objects           |
+--------------------+-------------------------------------------------------+
| `MWEBV`            | Simulated Milky Way extinction                        |
+--------------------+-------------------------------------------------------+
| `PTROBS_MIN`       | The row number (index - 1) in the corresponding       |
|                    | photometry table where data for the given object      |
|                    | starts                                                |
+--------------------+-------------------------------------------------------+
| `PTROBS_MAX`       | The row number (index - 1) in the corresponding       |
|                    | photometry table where data for the given object      |
|                    | starts                                                |
+--------------------+-------------------------------------------------------+
| `SIM_MODEL_NAME`   | Name of the model used to simulate the light-curve    |
+--------------------+-------------------------------------------------------+

+------------------------+---------------------------------------------------+
| Photometry File Column | Value Description                                 |
+========================+===================================================+
| `MJD`                  | Date of the observation                           |
+------------------------+---------------------------------------------------+
| `FLT`                  | The observed filter                               |
+------------------------+---------------------------------------------------+
| `FIELD`                | The field of the observation                      |
+------------------------+---------------------------------------------------+
| `PHOTFLAG`             | Either `0` (non-detection), `4096` (detection),   |
|                        | or `6144` (first trigger)                         |
+------------------------+---------------------------------------------------+
| `PHOTPROB`             |                                                   |
+------------------------+---------------------------------------------------+
| `FLUXCAL`              |                                                   |
+------------------------+---------------------------------------------------+
| `FLUXCALERR`           |                                                   |
+------------------------+---------------------------------------------------+
| `PSF_SIG1`             |                                                   |
+------------------------+---------------------------------------------------+
| `ZEROPT`               | The photometric zero point (`27.5`)               |
+------------------------+---------------------------------------------------+
| `SIM_MAGOBS`           | The simulated magnitude of the observations       |
+------------------------+---------------------------------------------------+