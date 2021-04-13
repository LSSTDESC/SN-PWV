"""The ``cov_utils`` module extends ``pandas.DataFrame`` objects to include
covariance calculations relevant to processing supernova fit results.

Usage Example
-------------

The ``cov_utils`` accessor provides a constructor method for converting
covariance data from ``numpy`` type objects into a ``pandas.DataFrame``
instance:

.. doctest::

   >>> import numpy as np
   >>> import pandas as pd
   >>> import snat_sim

   >>> parameter_names = ['z', 't0', 'x0', 'x1', 'c']
   >>> example_covariance = np.array([
   ...     [ 2.19e-04,  9.70e-04,  1.46e-09, -9.15e-04, -3.38e-04],
   ...     [ 9.70e-04,  1.75e-01, -2.75e-08,  4.74e-02, -6.88e-04],
   ...     [ 1.46e-09, -2.75e-08,  1.53e-13, -7.58e-08, -9.63e-09],
   ...     [-9.15e-04,  4.74e-02, -7.58e-08,  1.04e-01,  2.85e-03],
   ...     [-3.38e-04, -6.88e-04, -9.63e-09,  2.85e-03,  1.33e-03]
   ... ])
   >>> covariance_df = pd.DataFrame.cov_utils.from_array(example_covariance, parameter_names)
   >>> covariance_df
                  z            t0            x0            x1             c
   z   2.190000e-04  9.700000e-04  1.460000e-09 -9.150000e-04 -3.380000e-04
   t0  9.700000e-04  1.750000e-01 -2.750000e-08  4.740000e-02 -6.880000e-04
   x0  1.460000e-09 -2.750000e-08  1.530000e-13 -7.580000e-08 -9.630000e-09
   x1 -9.150000e-04  4.740000e-02 -7.580000e-08  1.040000e-01  2.850000e-03
   c  -3.380000e-04 -6.880000e-04 -9.630000e-09  2.850000e-03  1.330000e-03

From a given dataframe, you can easily extract a subset of the covariance data
using the ``subcovariance`` method:

.. doctest::

   >>> covariance_df.cov_utils.subcovariance(['x1', 'c'])
            x1        c
   x1  0.10400  0.00285
   c   0.00285  0.00133

Module Docs
-----------
"""

from typing import Collection, List, cast, overload
from warnings import warn

import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor('cov_utils')
class CovarianceAccessor:
    """Pandas DataFrame accessor for covariance calculations"""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Extends ``pandas`` support for time series data

        DO NOT USE THIS CLASS DIRECTLY! This class is registered as a pandas accessor.
        See the module level usage example for more information.

        Args:
            pandas_obj: Dataframe representing the covariance matrix
        """

        self._obj = pandas_obj.copy()

    @classmethod
    def from_array(
            cls, covArray: np.ndarray, paramNames: Collection[str] = None, normalized: bool = False
    ) -> pd.DataFrame:
        """Instantiates a Dataframe covariance matrix using data from a `numpy.ndarray` object.

        If paramNames is not None, then the dataframe is indexed by the
        parameter names, and has columns corresponding to the parameter names
        enabling easy access by index or names.

        Args:
            covArray: Array of the covariance
            paramNames: Collection of strings
            normalized: Whether to return the normalized covariance matrix

        Returns:
            A `pandas.DataFrame` with column names and indexes given by the parameter
            names. If paramNames is None, the return is a DataFrame with indexes and
            column names chosen by pandas.
        """

        length, width = np.shape(covArray)
        if length != width:  # Check for the covariance matrix being square, not checking for symmetry
            raise ValueError('The covariance matrix is not square; length!=width')

        if paramNames is not None:
            if len(paramNames) != width:
                raise ValueError('The number of parameters must match the length of the covariance matrix')

            cov = pd.DataFrame(covArray, columns=paramNames, index=paramNames)

        else:
            cov = pd.DataFrame(covArray)

        if not normalized:
            return cov

        # normalize if requested
        stds = cov.values.diagonal()
        for i, col in enumerate(cov.columns):
            cov[col] = cov[col] / stds[i]

        for i in range(len(cov)):
            cov.iloc[i] = cov.iloc[i] / stds[i]

        return cov

    @overload
    def log_covariance(self, paramName: int, paramValue: float, factor: float = 1.) -> np.ndarray:
        ...  # pragma: no cover

    @overload
    def log_covariance(self, paramName: str, paramValue: float, factor: float = 1.) -> pd.DataFrame:
        ...  # pragma: no cover

    def log_covariance(self, paramName, paramValue, factor=1.):
        """
        Covariance of the parameters with parameter paramName replaced by
        factor * np.log(param) everywhere, and its true value is paramValue,
        assuming linear propagation

        The ``factor`` parameter is used to scale the logarithm. For example,
        if the relevant transformation is going from ``f`` to ``-2.5 * log10(f)``,
        the factor should be ``-2.5 / np.log(10)``

        Args:
            paramName: Parameter name or integer index specifying the position of the desired variable
            paramValue: True or estimated value of the variable itself
            factor: Factor multiplying the natural logarithm
        """

        covariance_df = self._obj
        if isinstance(paramName, int):
            cov = covariance_df.values
            cov[:, paramName] = factor * cov[:, paramName] / paramValue
            cov[paramName, :] = factor * cov[paramName, :] / paramValue
            warn('Parameter name specified as index. Returning covariance as numpy array.')
            return cov

        covariance_df[paramName] = factor * covariance_df[paramName] / paramValue
        covariance_df.loc[paramName] = factor * covariance_df.loc[paramName] / paramValue

        return covariance_df

    def subcovariance(self, paramList: List[str]) -> pd.DataFrame:
        """Returns the covariance of a subset of parameters in a covariance dataFrame.

        The set of parameters in paramList must be a subset of the columns and
        indices of covariance.

        Args:
            paramList: List of parameters for which the subCovariance matrix is desired

        Returns:
            The covariance of the given parameters
        """

        return self._obj.loc[paramList, paramList]

    def expAVsquare(self, A: np.array) -> float:
        """The expectation of ``(A^T * V) ** 2`` where A is a constant vector and V is
        a random vector ``V ~ N(0., covV)`` by computing ``A^T * covV * A``

        Args:
            A: Vector of constants.

        Returns:
            The variance as a scalar value
        """

        va = np.sum(self._obj * A, axis=1)
        var = np.sum(A * va, axis=0)
        return cast(float, var)
