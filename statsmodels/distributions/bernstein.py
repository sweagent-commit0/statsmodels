"""
Created on Wed Feb 17 15:35:23 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import _Grid, cdf2prob_grid, prob2cdf_grid, _eval_bernstein_dd, _eval_bernstein_2d, _eval_bernstein_1d

class BernsteinDistribution:
    """Distribution based on Bernstein Polynomials on unit hypercube.

    Parameters
    ----------
    cdf_grid : array_like
        cdf values on a equal spaced grid of the unit hypercube [0, 1]^d.
        The dimension of the arrays define how many random variables are
        included in the multivariate distribution.

    Attributes
    ----------
    cdf_grid : grid of cdf values
    prob_grid : grid of cell or bin probabilities
    k_dim : (int) number of components, dimension of random variable
    k_grid : (tuple) shape of cdf_grid
    k_grid_product : (int) total number of bins in grid
    _grid : Grid instance with helper methods and attributes
    """

    def __init__(self, cdf_grid):
        self.cdf_grid = cdf_grid = np.asarray(cdf_grid)
        self.k_dim = cdf_grid.ndim
        self.k_grid = cdf_grid.shape
        self.k_grid_product = np.prod([i - 1 for i in self.k_grid])
        self._grid = _Grid(self.k_grid)

    @classmethod
    def from_data(cls, data, k_bins):
        """Create distribution instance from data using histogram binning.

        Classmethod to construct a distribution instance.

        Parameters
        ----------
        data : array_like
            Data with observation in rows and random variables in columns.
            Data can be 1-dimensional in the univariate case.
        k_bins : int or list
            Number or edges of bins to be used in numpy histogramdd.
            If k_bins is a scalar int, then the number of bins of each
            component will be equal to it.

        Returns
        -------
        Instance of a Bernstein distribution
        """
        pass

    def cdf(self, x):
        """cdf values evaluated at x.

        Parameters
        ----------
        x : array_like
            Points of multivariate random variable at which cdf is evaluated.
            This can be a single point with length equal to the dimension of
            the random variable, or two dimensional with points (observations)
            in rows and random variables in columns.
            In the univariate case, a 1-dimensional x will be interpreted as
            different points for evaluation.

        Returns
        -------
        pdf values

        Notes
        -----
        Warning: 2-dim x with many points can be memory intensive because
        currently the bernstein polynomials will be evaluated in a fully
        vectorized computation.
        """
        pass

    def pdf(self, x):
        """pdf values evaluated at x.

        Parameters
        ----------
        x : array_like
            Points of multivariate random variable at which pdf is evaluated.
            This can be a single point with length equal to the dimension of
            the random variable, or two dimensional with points (observations)
            in rows and random variables in columns.
            In the univariate case, a 1-dimensional x will be interpreted as
            different points for evaluation.

        Returns
        -------
        cdf values

        Notes
        -----
        Warning: 2-dim x with many points can be memory intensive because
        currently the bernstein polynomials will be evaluated in a fully
        vectorized computation.
        """
        pass

    def get_marginal(self, idx):
        """Get marginal BernsteinDistribution.

        Parameters
        ----------
        idx : int or list of int
            Index or indices of the component for which the marginal
            distribution is returned.

        Returns
        -------
        BernsteinDistribution instance for the marginal distribution.
        """
        pass

    def rvs(self, nobs):
        """Generate random numbers from distribution.

        Parameters
        ----------
        nobs : int
            Number of random observations to generate.
        """
        pass

class BernsteinDistributionBV(BernsteinDistribution):
    pass

class BernsteinDistributionUV(BernsteinDistribution):
    pass