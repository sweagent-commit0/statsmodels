"""
Created on Thu Feb 11 09:19:30 2021

Author: Josef Perktold
License: BSD-3

"""
import warnings
import numpy as np
from scipy import interpolate, stats

class _Grid:
    """Create Grid values and indices, grid in [0, 1]^d

    This class creates a regular grid in a d dimensional hyper cube.

    Intended for internal use, implementation might change without warning.


    Parameters
    ----------
    k_grid : tuple or array_like
        number of elements for axes, this defines k_grid - 1 equal sized
        intervals of [0, 1] for each axis.
    eps : float
        If eps is not zero, then x values will be clipped to [eps, 1 - eps],
        i.e. to the interior of the unit interval or hyper cube.


    Attributes
    ----------
    k_grid : list of number of grid points
    x_marginal: list of 1-dimensional marginal values
    idx_flat: integer array with indices
    x_flat: flattened grid values,
        rows are grid points, columns represent variables or axis.
        ``x_flat`` is currently also 2-dim in the univariate 1-dim grid case.

    """

    def __init__(self, k_grid, eps=0):
        self.k_grid = k_grid
        x_marginal = [np.arange(ki) / (ki - 1) for ki in k_grid]
        idx_flat = np.column_stack(np.unravel_index(np.arange(np.prod(k_grid)), k_grid)).astype(float)
        x_flat = idx_flat / idx_flat.max(0)
        if eps != 0:
            x_marginal = [np.clip(xi, eps, 1 - eps) for xi in x_marginal]
            x_flat = np.clip(x_flat, eps, 1 - eps)
        self.x_marginal = x_marginal
        self.idx_flat = idx_flat
        self.x_flat = x_flat

def prob2cdf_grid(probs):
    """Cumulative probabilities from cell provabilites on a grid

    Parameters
    ----------
    probs : array_like
        Rectangular grid of cell probabilities.

    Returns
    -------
    cdf : ndarray
        Grid of cumulative probabilities with same shape as probs.
    """
    pass

def cdf2prob_grid(cdf, prepend=0):
    """Cell probabilities from cumulative probabilities on a grid.

    Parameters
    ----------
    cdf : array_like
        Grid of cumulative probabilities with same shape as probs.

    Returns
    -------
    probs : ndarray
        Rectangular grid of cell probabilities.

    """
    pass

def average_grid(values, coords=None, _method='slicing'):
    """Compute average for each cell in grid using endpoints

    Parameters
    ----------
    values : array_like
        Values on a grid that will average over corner points of each cell.
    coords : None or list of array_like
        Grid coordinates for each axis use to compute volumne of cell.
        If None, then averaged values are not rescaled.
    _method : {"slicing", "convolve"}
        Grid averaging is implemented using numpy "slicing" or using
        scipy.signal "convolve".

    Returns
    -------
    Grid with averaged cell values.
    """
    pass

def nearest_matrix_margins(mat, maxiter=100, tol=1e-08):
    """nearest matrix with uniform margins

    Parameters
    ----------
    mat : array_like, 2-D
        Matrix that will be converted to have uniform margins.
        Currently, `mat` has to be two dimensional.
    maxiter : in
        Maximum number of iterations.
    tol : float
        Tolerance for convergence, defined for difference between largest and
        smallest margin in each dimension.

    Returns
    -------
    ndarray, nearest matrix with uniform margins.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.

    changed in 0.14 to support k_dim > 2.


    """
    pass

def _rankdata_no_ties(x):
    """rankdata without ties for 2-d array

    This is a simplified version for ranking data if there are no ties.
    Works vectorized across columns.

    See Also
    --------
    scipy.stats.rankdata

    """
    pass

def frequencies_fromdata(data, k_bins, use_ranks=True):
    """count of observations in bins (histogram)

    currently only for bivariate data

    Parameters
    ----------
    data : array_like
        Bivariate data with observations in rows and two columns. Binning is
        in unit rectangle [0, 1]^2. If use_rank is False, then data should be
        in unit interval.
    k_bins : int
        Number of bins along each dimension in the histogram
    use_ranks : bool
        If use_rank is True, then data will be converted to ranks without
        tie handling.

    Returns
    -------
    bin counts : ndarray
        Frequencies are the number of observations in a given bin.
        Bin counts are a 2-dim array with k_bins rows and k_bins columns.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    pass

def approx_copula_pdf(copula, k_bins=10, force_uniform=True, use_pdf=False):
    """Histogram probabilities as approximation to a copula density.

    Parameters
    ----------
    copula : instance
        Instance of a copula class. Only the ``pdf`` method is used.
    k_bins : int
        Number of bins along each dimension in the approximating histogram.
    force_uniform : bool
        If true, then the pdf grid will be adjusted to have uniform margins
        using `nearest_matrix_margin`.
        If false, then no adjustment is done and the margins may not be exactly
        uniform.
    use_pdf : bool
        If false, then the grid cell probabilities will be computed from the
        copula cdf.
        If true, then the density, ``pdf``, is used and cell probabilities
        are approximated by averaging the pdf of the cell corners. This is
        only useful if the cdf is not available.

    Returns
    -------
    bin probabilites : ndarray
        Probability that random variable falls in given bin. This corresponds
        to a discrete distribution, and is not scaled to bin size to form a
        piecewise uniform, histogram density.
        Bin probabilities are a k-dim array with k_bins segments in each
        dimensionrows.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    pass

def _eval_bernstein_1d(x, fvals, method='binom'):
    """Evaluate 1-dimensional bernstein polynomial given grid of values.

    experimental, comparing methods

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.
    method: "binom", "beta" or "bpoly"
        Method to construct Bernstein polynomial basis, used for comparison
        of parameterizations.

        - "binom" uses pmf of Binomial distribution
        - "beta" uses pdf of Beta distribution
        - "bpoly" uses one interval in scipy.interpolate.BPoly

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    pass

def _eval_bernstein_2d(x, fvals):
    """Evaluate 2-dimensional bernstein polynomial given grid of values

    experimental

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    pass

def _eval_bernstein_dd(x, fvals):
    """Evaluate d-dimensional bernstein polynomial given grid of valuesv

    experimental

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    pass

def _ecdf_mv(data, method='seq', use_ranks=True):
    """
    Multivariate empiricial distribution function, empirical copula


    Notes
    -----
    Method "seq" is faster than method "brute", but supports mainly bivariate
    case. Speed advantage of "seq" is increasing in number of observations
    and decreasing in number of variables.
    (see Segers ...)

    Warning: This does not handle ties. The ecdf is based on univariate ranks
    without ties. The assignment of ranks to ties depends on the sorting
    algorithm and the initial ordering of the data.

    When the original data is used instead of ranks, then method "brute"
    computes the correct ecdf counts even in the case of ties.

    """
    pass