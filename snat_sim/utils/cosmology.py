import inspect
from typing import List, Union

import numpy as np
import pandas as pd
from astropy.cosmology import FlatwCDM
from iminuit import Minuit

FloatOrArray = Union[float, np.ndarray]


@pd.api.extensions.register_dataframe_accessor("snat_sim")
class CosmologyAccessor:
    """Chi-squared minimizer for fitting a cosmology to pipeline results"""

    def __init__(self, pandas_obj: pd.Series) -> None:
        self.data = pandas_obj

    def calc_distmod(self, abs_mag: float) -> pd.Series:
        """Return the distance modulus for an assumed absolute magnitude

        Args:
            abs_mag: The B-band absolute magnitude

        Returns:
            The distance modulus
        """

        return self.data['mb'] - abs_mag

    # noinspection PyPep8Naming
    def chisq(self, H0: float, Om0: float, abs_mag: float, w0: float, alpha: float, beta: float) -> float:
        """Calculate the chi-squared for given cosmological parameters

        Args:
            H0: Hubble constant
            Om0: Matter density
            abs_mag: SNe Ia intrinsic peak magnitude
            w0: Dark matter equation of state
            alpha: Stretch correction nuisance parameter
            beta: Color correction nuisance parameter

        Returns:
            The chi-squared of the given cosmology
        """

        measured_mu = self.calc_distmod(abs_mag) + alpha * self.data['x1'] - beta * self.data['c']

        cosmology = FlatwCDM(H0=H0, Om0=Om0, w0=w0)
        modeled_mu = cosmology.distmod(self.data['z']).value
        return np.sum(((measured_mu - modeled_mu) ** 2) / (self.data['mb_err'] ** 2))

    def chisq_grid(
            self,
            H0: FloatOrArray,
            Om0: FloatOrArray,
            abs_mag: FloatOrArray,
            w0: FloatOrArray,
            alpha: float,
            beta: float
    ) -> np.ndarray:
        """Calculate the chi-squared on a grid of cosmological parameters

        Arguments are automatically repeated along the grid so that the
        dimensions of each array match.

        Args:
            H0: Hubble constant
            Om0: Matter density
            abs_mag: SNe Ia intrinsic peak magnitude
            w0: Dark matter equation of state
            alpha: Stretch correction nuisance parameter
            beta: Color correction nuisance parameter

        Returns:
            An array of chi-squared values
        """

        new_args = self._match_argument_dimensions(H0, Om0, abs_mag, w0, alpha, beta)
        return np.vectorize(self.chisq)(*new_args)

    @staticmethod
    def _match_argument_dimensions(*args: FloatOrArray) -> list:
        """Reshape arguments so they match the shape of the argument with the
        most dimensions.

        Args:
            *args: Values to cast onto the grid

        Returns:
            A list with each argument cast to it's new shape
        """

        # Get the shape of the argument with the most dimensions
        grid_shape = np.shape(args[np.argmax([np.ndim(arg) for arg in args])])

        # Reshape each argument to match the dimensions from above
        return [np.full(grid_shape, arg) for arg in args]

    def minimize(self, **kwargs) -> Minuit:
        """Fit cosmology to the instantiated data

        Kwargs:
            Accepts any iminuit style keyword arguments for parameters
              ``H0``, ``Om0``, ``abs_mag``, and ``w0``.

        Returns:
            Optimized Minuit object
        """

        minimizer = Minuit(self.chisq, **kwargs)
        minimizer.migrad()
        return minimizer

    def minimize_mc(
            self, samples: int, n: int = None, frac: float = None, statistic: callable = None, **kwargs
    ) -> List[Minuit]:
        """Fit cosmology to the instantiated data using monte carlo resampling

        Args:
            samples: Number of samples to draw
            n: Size of each sample. Cannot be used with ``frac``
            frac: Fraction of data to include in each sample. Cannot be used with ``size``
            statistic: Optionally apply a statistic to the returned values
            **kwargs: Accepts any iminuit style keyword arguments for parameters
              ``H0``, ``Om0``, ``abs_mag``, and ``w0``.

        Returns:
            List of optimized Minuit object or a dictionary of the applies statistic to those values
        """

        if statistic:
            samples = [self.data.sample(n=n, frac=frac).snat_sim.minimize(**kwargs).np_values() for _ in range(samples)]
            stat_val = statistic(samples)

            # Create a dictionary mapping the argument name to the applies statistic
            arg_names = inspect.getfullargspec(self.chisq).args
            samples = dict(zip(arg_names[1:], stat_val))  # first argument is self, so drop it

        else:
            samples = [self.data.sample(n=n, frac=frac).snat_sim.minimize(**kwargs) for _ in range(samples)]

        return samples
