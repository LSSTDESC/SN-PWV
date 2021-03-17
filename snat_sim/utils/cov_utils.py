"""A number of utility functions to conveniently deal with covariances

Ported from https://github.com/rbiswas4/AnalyzeSN/ under the MIT License.
For more information see the provenance section of the documentation.
"""

from typing import Collection, List, cast, overload

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
        """
        converts a covariance matrix from a `numpy.ndarray` to a
        `pandas.DataFrame`. If paramNames is not None, then the dataframe
        is indexed by the parameter names, and has columns corresponding
        to the parameter names enabling easy access by index or names.

        Args:
            covArray: Array of the covariance
            paramNames: Collection of strings
            normalized: Whether to return the normalized covariance matrix

        Returns:
            A `pandas.DataFrame` with column names and indexes given by the parameter
            names. If paramNames is None, the return is a DataFrame with indexes and
            column names chosen by pandas.
        """

        l, w = np.shape(covArray)
        if l != w:  # Check for the covariance matrix being square, not checking for symmetry
            raise ValueError('The covariance matrix is not square; length!=width')

        if paramNames is not None:
            if len(paramNames) != w:
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
        ...

    @overload
    def log_covariance(self, paramName: str, paramValue: float, factor: float = 1.) -> pd.DataFrame:
        ...

    def log_covariance(self, paramName, paramValue, factor=1.):
        """
        Covariance of the parameters with parameter paramName replaced by
        factor * np.log(param) everywhere, and its true value is paramValue,
        assuming linear propagation

        The ``factor`` parameter is used to define the logarithm. For example,
        if the relevant transformation is going from 'f' to -2.5 log10(f),
        the factor should be -2.5 /np.log(10)

        Args:
            paramName: Integer or parameter name specifying the position of the variable whose logarithm must be taken
            paramValue: True/estimated value of the variable itself
            factor: Factor multiplying the natural logarithm
        """

        covariance_df = self._obj
        if isinstance(paramName, np.int):
            cov = covariance_df.values
            cov[:, paramName] = factor * cov[:, paramName] / paramValue
            cov[paramName, :] = factor * cov[paramName, :] / paramValue
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
        """
        Return the expectation of (A^T V)^2 where A is a constant vector and V is
        a random vector V ~ N(0., covV) by computing A^T * covV * A

        Args:
            A: Vector of constants.

        Returns:
            The variance as a scalar value
        """

        va = np.sum(self._obj * A, axis=1)
        var = np.sum(A * va, axis=0)
        return cast(float, var)
