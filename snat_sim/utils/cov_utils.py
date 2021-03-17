"""A number of utility functions to conveniently deal with covariances

Ported from https://github.com/rbiswas4/AnalyzeSN/ under the MIT License.
For more information see the provenance section of the documentation.
"""

from typing import Collection, List, Union, cast

import numpy as np
import pandas as pd


def expAVsquare(covV: np.array, A: np.array) -> float:
    """
    Return the expectation of (A^T V)^2 where A is a constant vector and V is
    a random vector V ~ N(0., covV) by computing A^T * covV * A

    Args:
        covV: mandatory
        A: Vector of constants.

    Returns:
        The variance as a scalar value
    """

    va = np.sum(covV * A, axis=1)
    var = np.sum(A * va, axis=0)
    return cast(float, var)


def log_covariance(covariance: pd.DataFrame, paramName: Union[int, str], paramValue: float, factor: float = 1.):
    """
    Covariance of the parameters with parameter paramName replaced by
    factor * np.log(param) everywhere, and its true value is paramValue,
    assuming linear propagation

    The ``factor`` parameter is used to define the logarithm. For example,
    if the relevant transformation is going from 'f' to -2.5 log10(f),
    the factor should be -2.5 /np.log(10)

    Args:
        covariance:Dataframe representing the covariance matrix
        paramName : Integer or parameter name specifying the position of the variable whose logarithm must be taken
        paramValue : float, mandatory
        true/estimated value of the variable itself
        factor: Factor multiplying the natural logarithm.
    """

    if isinstance(paramName, np.int):
        cov = covariance.values
        cov[:, paramName] = factor * cov[:, paramName] / paramValue
        cov[paramName, :] = factor * cov[paramName, :] / paramValue
        return cov

    covariance[paramName] = factor * covariance[paramName] / paramValue
    covariance.loc[paramName] = factor * covariance.loc[paramName] / paramValue

    return covariance


def subcovariance(covariance: pd.DataFrame, paramList: List[str], array: bool = False):
    """Returns the covariance of a subset of parameters in a covariance dataFrame.

    Args:
        covariance: representing square covariance matrix with parameters as column names, and index as returned by covariance
        paramList:
            list of parameters for which the subCovariance matrix is desired.
            The set of parameters in paramList must be a subset of the columns
            and indices of covariance
        array: if true, return `numpy.ndarray`, if False return `pandas.DataFrame`

    Returns:

    """

    df = covariance.loc[paramList, paramList]
    if array:
        return df.values

    return df


def covariance(covArray: np.ndarray, paramNames: Collection[str] = None, normalized: bool = False) -> pd.DataFrame:
    """
    converts a covariance matrix in `numpy.ndarray` to a
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
    # Check for the covariance matrix being square, not checking for symmetry
    if l != w:
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

# From the original source, but not needed for our purposes here
# def generateCov(dims: int, seed: Optional[int] = None, low: float = -0.5, high: float = 0.5) -> np.array:
#     """
#     Generate a 2D semi-positive definite matrix of size dimsXdims. While
#     this will create different random matrices, the exact distribution of
#     the matrices has not been checked.
#
#     Args:
#         dims: Size of the matrix
#         seed: Optionally sets the seed of the random number generator.
#         low: Entries are x * y, and the smallest value for x, or y is low
#         high: Entries are x * y, and the largest value for x, or y is high
#
#     Returns:
#         THe 2D matrix as a numpy array
#     """
#
#     if seed is not None:
#         np.random.seed(seed)
#
#     x = np.random.uniform(low, high, size=dims)
#     y = np.random.uniform(low, high, size=dims)
#     m = np.outer(x, y)
#     return np.dot(m, m.transpose())
