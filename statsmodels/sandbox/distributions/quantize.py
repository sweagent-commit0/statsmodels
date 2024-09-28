"""Quantizing a continuous distribution in 2d

Author: josef-pktd
"""
from statsmodels.compat.python import lmap
import numpy as np

def prob_bv_rectangle(lower, upper, cdf):
    """helper function for probability of a rectangle in a bivariate distribution

    Parameters
    ----------
    lower : array_like
        tuple of lower integration bounds
    upper : array_like
        tuple of upper integration bounds
    cdf : callable
        cdf(x,y), cumulative distribution function of bivariate distribution


    how does this generalize to more than 2 variates ?
    """
    pass

def prob_mv_grid(bins, cdf, axis=-1):
    """helper function for probability of a rectangle grid in a multivariate distribution

    how does this generalize to more than 2 variates ?

    bins : tuple
        tuple of bin edges, currently it is assumed that they broadcast
        correctly

    """
    pass

def prob_quantize_cdf(binsx, binsy, cdf):
    """quantize a continuous distribution given by a cdf

    Parameters
    ----------
    binsx : array_like, 1d
        binedges

    """
    pass

def prob_quantize_cdf_old(binsx, binsy, cdf):
    """quantize a continuous distribution given by a cdf

    old version without precomputing cdf values

    Parameters
    ----------
    binsx : array_like, 1d
        binedges

    """
    pass
if __name__ == '__main__':
    from numpy.testing import assert_almost_equal
    unif_2d = lambda x, y: x * y
    assert_almost_equal(prob_bv_rectangle([0, 0], [1, 0.5], unif_2d), 0.5, 14)
    assert_almost_equal(prob_bv_rectangle([0, 0], [0.5, 0.5], unif_2d), 0.25, 14)
    arr1b = np.array([[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]])
    arr1a = prob_quantize_cdf(np.linspace(0, 1, 6), np.linspace(0, 1, 5), unif_2d)
    assert_almost_equal(arr1a, arr1b, 14)
    arr2b = np.array([[0.25], [0.25], [0.25], [0.25]])
    arr2a = prob_quantize_cdf(np.linspace(0, 1, 5), np.linspace(0, 1, 2), unif_2d)
    assert_almost_equal(arr2a, arr2b, 14)
    arr3b = np.array([[0.25, 0.25, 0.25, 0.25]])
    arr3a = prob_quantize_cdf(np.linspace(0, 1, 2), np.linspace(0, 1, 5), unif_2d)
    assert_almost_equal(arr3a, arr3b, 14)