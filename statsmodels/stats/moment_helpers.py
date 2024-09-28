"""helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix


Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy.special import comb

def mc2mnc(mc):
    """convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    pass

def mnc2mc(mnc, wmean=True):
    """convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    pass

def cum2mc(kappa):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    """
    pass

def mnc2cum(mnc):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    """
    pass

def mc2cum(mc):
    """
    just chained because I have still the test case
    """
    pass

def mvsk2mc(args):
    """convert mean, variance, skew, kurtosis to central moments"""
    pass

def mvsk2mnc(args):
    """convert mean, variance, skew, kurtosis to non-central moments"""
    pass

def mc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis"""
    pass

def mnc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis
    """
    pass

def cov2corr(cov, return_std=False):
    """
    convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that
    division is defined elementwise. np.ma.array and np.matrix are allowed.
    """
    pass

def corr2cov(corr, std):
    """
    convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.
    """
    pass

def se_cov(cov):
    """
    get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    Parameters
    ----------
    cov : array_like, square
        covariance matrix

    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov
    """
    pass