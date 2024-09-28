"""
BDS test for IID time series

References
----------

Broock, W. A., J. A. Scheinkman, W. D. Dechert, and B. LeBaron. 1996.
"A Test for Independence Based on the Correlation Dimension."
Econometric Reviews 15 (3): 197-235.

Kanzler, Ludwig. 1999.
"Very Fast and Correctly Sized Estimation of the BDS Statistic".
SSRN Scholarly Paper ID 151669. Rochester, NY: Social Science Research Network.

LeBaron, Blake. 1997.
"A Fast Algorithm for the BDS Statistic."
Studies in Nonlinear Dynamics & Econometrics 2 (2) (January 1).
"""
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like

def distance_indicators(x, epsilon=None, distance=1.5):
    """
    Calculate all pairwise threshold distance indicators for a time series

    Parameters
    ----------
    x : 1d array
        observations of time series for which heaviside distance indicators
        are calculated
    epsilon : scalar, optional
        the threshold distance to use in calculating the heaviside indicators
    distance : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it

    Returns
    -------
    indicators : 2d array
        matrix of distance threshold indicators

    Notes
    -----
    Since this can be a very large matrix, use np.int8 to save some space.
    """
    pass

def correlation_sum(indicators, embedding_dim):
    """
    Calculate a correlation sum

    Useful as an estimator of a correlation integral

    Parameters
    ----------
    indicators : ndarray
        2d array of distance threshold indicators
    embedding_dim : int
        embedding dimension

    Returns
    -------
    corrsum : float
        Correlation sum
    indicators_joint
        matrix of joint-distance-threshold indicators
    """
    pass

def correlation_sums(indicators, max_dim):
    """
    Calculate all correlation sums for embedding dimensions 1:max_dim

    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    max_dim : int
        maximum embedding dimension

    Returns
    -------
    corrsums : ndarray
        Correlation sums
    """
    pass

def _var(indicators, max_dim):
    """
    Calculate the variance of a BDS effect

    Parameters
    ----------
    indicators : ndarray
        2d array of distance threshold indicators
    max_dim : int
        maximum embedding dimension

    Returns
    -------
    variances : float
        Variance of BDS effect
    """
    pass

def bds(x, max_dim=2, epsilon=None, distance=1.5):
    """
    BDS Test Statistic for Independence of a Time Series

    Parameters
    ----------
    x : ndarray
        Observations of time series for which bds statistics is calculated.
    max_dim : int
        The maximum embedding dimension.
    epsilon : {float, None}, optional
        The threshold distance to use in calculating the correlation sum.
    distance : float, optional
        Specifies the distance multiplier to use when computing the test
        statistic if epsilon is omitted.

    Returns
    -------
    bds_stat : float
        The BDS statistic.
    pvalue : float
        The p-values associated with the BDS statistic.

    Notes
    -----
    The null hypothesis of the test statistic is for an independent and
    identically distributed (i.i.d.) time series, and an unspecified
    alternative hypothesis.

    This test is often used as a residual diagnostic.

    The calculation involves matrices of size (nobs, nobs), so this test
    will not work with very long datasets.

    Implementation conditions on the first m-1 initial values, which are
    required to calculate the m-histories:
    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})
    """
    pass