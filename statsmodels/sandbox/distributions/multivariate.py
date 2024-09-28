"""Multivariate Distribution

Probability of a multivariate t distribution

Now also mvstnormcdf has tests against R mvtnorm

Still need non-central t, extra options, and convenience function for
location, scale version.

Author: Josef Perktold
License: BSD (3-clause)

Reference:
Genz and Bretz for formula

"""
import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi
from .extras import mvstdnormcdf
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln

def chi2_pdf(self, x, df):
    """pdf of chi-square distribution"""
    pass

def mvstdtprob(a, b, R, df, ieps=1e-05, quadkwds=None, mvstkwds=None):
    """
    Probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.
    """
    pass

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    """generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    """
    pass
if __name__ == '__main__':
    corr = np.asarray([[1.0, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
    corr_indep = np.asarray([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    corr_equal = np.asarray([[1.0, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    R = corr_equal
    a = np.array([-np.inf, -np.inf, -100.0])
    a = np.array([-0.96, -0.96, -0.96])
    b = np.array([0.0, 0.0, 0.0])
    b = np.array([0.96, 0.96, 0.96])
    a[:] = -1
    b[:] = 3
    df = 10.0
    sqrt_df = np.sqrt(df)
    print(mvstdnormcdf(a, b, corr, abseps=1e-06))
    print((stats.t.cdf(b[0], df) - stats.t.cdf(a[0], df)) ** 3)
    s = 1
    print(mvstdnormcdf(s * a / sqrt_df, s * b / sqrt_df, R))
    df = 4
    print(mvstdtprob(a, b, R, df))
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    print(multivariate_t_rvs([10.0, 20.0], S, 2, 5))
    nobs = 10000
    rvst = multivariate_t_rvs([10.0, 20.0], S, 2, nobs)
    print(np.sum((rvst < [10.0, 20.0]).all(1), 0) * 1.0 / nobs)
    print(mvstdtprob(-np.inf * np.ones(2), np.zeros(2), R[:2, :2], 2))
    '\n        > lower <- -1\n        > upper <- 3\n        > df <- 4\n        > corr <- diag(3)\n        > delta <- rep(0, 3)\n        > pmvt(lower=lower, upper=upper, delta=delta, df=df, corr=corr)\n        [1] 0.5300413\n        attr(,"error")\n        [1] 4.321136e-05\n        attr(,"msg")\n        [1] "Normal Completion"\n        > (pt(upper, df) - pt(lower, df))**3\n        [1] 0.4988254\n\n    '