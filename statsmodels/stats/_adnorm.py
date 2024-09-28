"""
Created on Sun Sep 25 21:23:38 2011

Author: Josef Perktold and Scipy developers
License : BSD-3
"""
import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like, bool_like, int_like

def anderson_statistic(x, dist='norm', fit=True, params=(), axis=0):
    """
    Calculate the Anderson-Darling a2 statistic.

    Parameters
    ----------
    x : array_like
        The data to test.
    dist : {'norm', callable}
        The assumed distribution under the null of test statistic.
    fit : bool
        If True, then the distribution parameters are estimated.
        Currently only for 1d data x, except in case dist='norm'.
    params : tuple
        The optional distribution parameters if fit is False.
    axis : int
        If dist is 'norm' or fit is False, then data can be an n-dimensional
        and axis specifies the axis of a variable.

    Returns
    -------
    {float, ndarray}
        The Anderson-Darling statistic.
    """
    pass

def normal_ad(x, axis=0):
    """
    Anderson-Darling test for normal distribution unknown mean and variance.

    Parameters
    ----------
    x : array_like
        The data array.
    axis : int
        The axis to perform the test along.

    Returns
    -------
    ad2 : float
        Anderson Darling test statistic.
    pval : float
        The pvalue for hypothesis that the data comes from a normal
        distribution with unknown mean and variance.

    See Also
    --------
    statsmodels.stats.diagnostic.anderson_statistic
        The Anderson-Darling a2 statistic.
    statsmodels.stats.diagnostic.kstest_fit
        Kolmogorov-Smirnov test with estimated parameters for Normal or
        Exponential distributions.
    """
    pass