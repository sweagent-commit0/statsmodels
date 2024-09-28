"""Correlation and Covariance Structures

Created on Sat Dec 17 20:46:05 2011

Author: Josef Perktold
License: BSD-3


Reference
---------
quick reading of some section on mixed effects models in S-plus and of
outline for GEE.

"""
import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr

def corr_equi(k_vars, rho):
    """create equicorrelated correlation matrix with rho on off diagonal

    Parameters
    ----------
    k_vars : int
        number of variables, correlation matrix will be (k_vars, k_vars)
    rho : float
        correlation between any two random variables

    Returns
    -------
    corr : ndarray (k_vars, k_vars)
        correlation matrix

    """
    pass

def corr_ar(k_vars, ar):
    """create autoregressive correlation matrix

    This might be MA, not AR, process if used for residual process - check

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0


    """
    pass

def corr_arma(k_vars, ar, ma):
    """create arma correlation matrix

    converts arma to autoregressive lag-polynomial with k_var lags

    ar and arma might need to be switched for generating residual process

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0
    ma : array_like, 1d
        MA lag-polynomial

    """
    pass

def corr2cov(corr, std):
    """convert correlation matrix to covariance matrix

    Parameters
    ----------
    corr : ndarray, (k_vars, k_vars)
        correlation matrix
    std : ndarray, (k_vars,) or scalar
        standard deviation for the vector of random variables. If scalar, then
        it is assumed that all variables have the same scale given by std.

    """
    pass

def whiten_ar(x, ar_coefs, order):
    """
    Whiten a series of columns according to an AR(p) covariance structure.

    This drops the initial conditions (Cochran-Orcut ?)
    Uses loop, so for short ar polynomials only, use lfilter otherwise

    This needs to improve, option on method, full additional to conditional

    Parameters
    ----------
    x : array_like, (nobs,) or (nobs, k_vars)
        The data to be whitened along axis 0
    ar_coefs : ndarray
        coefficients of AR lag- polynomial,   TODO: ar or ar_coefs?
    order : int

    Returns
    -------
    x_new : ndarray
        transformed array
    """
    pass

def yule_walker_acov(acov, order=1, method='unbiased', df=None, inv=False):
    """
    Estimate AR(p) parameters from acovf using Yule-Walker equation.


    Parameters
    ----------
    acov : array_like, 1d
        auto-covariance
    order : int, optional
        The order of the autoregressive process.  Default is 1.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.

    Returns
    -------
    rho : ndarray
        The estimated autoregressive coefficients
    sigma
        TODO
    Rinv : ndarray
        inverse of the Toepliz matrix
    """
    pass

class ARCovariance:
    """
    experimental class for Covariance of AR process
    classmethod? staticmethods?
    """

    def __init__(self, ar=None, ar_coefs=None, sigma=1.0):
        if ar is not None:
            self.ar = ar
            self.ar_coefs = -ar[1:]
            self.k_lags = len(ar)
        elif ar_coefs is not None:
            self.arcoefs = ar_coefs
            self.ar = np.hstack(([1], -ar_coefs))
            self.k_lags = len(self.ar)