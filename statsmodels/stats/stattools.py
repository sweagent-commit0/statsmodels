"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions have not been formally tested.
"""
from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning

def durbin_watson(resids, axis=0):
    """
    Calculates the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \\sum_{t=2}^T((e_t - e_{t-1})^2)/\\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.
    """
    pass

def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    ----------
    resid : array_like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    pass

def jarque_bera(resids, axis=0):
    """
    The Jarque-Bera test of normality.

    Parameters
    ----------
    resids : array_like
        Data to test for normality. Usually regression model residuals that
        are mean 0.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    JB : {float, ndarray}
        The Jarque-Bera test statistic.
    JBpv : {float, ndarray}
        The pvalue of the test statistic.
    skew : {float, ndarray}
        Estimated skewness of the data.
    kurtosis : {float, ndarray}
        Estimated kurtosis of the data.

    Notes
    -----
    Each output returned has 1 dimension fewer than data

    The Jarque-Bera test statistic tests the null that the data is normally
    distributed against an alternative that the data follow some other
    distribution. The test statistic is based on two moments of the data,
    the skewness, and the kurtosis, and has an asymptotic :math:`\\chi^2_2`
    distribution.

    The test statistic is defined

    .. math:: JB = n(S^2/6+(K-3)^2/24)

    where n is the number of data points, S is the sample skewness, and K is
    the sample kurtosis of the data.
    """
    pass

def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the skewness measures are computed.  If `None`, the
        entire array is used.

    Returns
    -------
    sk1 : ndarray
          The standard skewness estimator.
    sk2 : ndarray
          Skewness estimator based on quartiles.
    sk3 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          absolute deviation.
    sk4 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          standard deviation.

    Notes
    -----
    The robust skewness measures are defined

    .. math::

        SK_{2}=\\frac{\\left(q_{.75}-q_{.5}\\right)
        -\\left(q_{.5}-q_{.25}\\right)}{q_{.75}-q_{.25}}

    .. math::

        SK_{3}=\\frac{\\mu-\\hat{q}_{0.5}}
        {\\hat{E}\\left[\\left|y-\\hat{\\mu}\\right|\\right]}

    .. math::

        SK_{4}=\\frac{\\mu-\\hat{q}_{0.5}}{\\hat{\\sigma}}

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    pass

def _kr3(y, alpha=5.0, beta=50.0):
    """
    KR3 estimator from Kim & White

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.
    alpha : float, optional
        Lower cut-off for measuring expectation in tail.
    beta :  float, optional
        Lower cut-off for measuring expectation in center.

    Returns
    -------
    kr3 : float
        Robust kurtosis estimator based on standardized lower- and upper-tail
        expected values

    Notes
    -----
    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    pass

def expected_robust_kurtosis(ab=(5.0, 50.0), dg=(2.5, 25.0)):
    """
    Calculates the expected value of the robust kurtosis measures in Kim and
    White assuming the data are normally distributed.

    Parameters
    ----------
    ab : iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure

    Returns
    -------
    ekr : ndarray, 4-element
        Contains the expected values of the 4 robust kurtosis measures

    Notes
    -----
    See `robust_kurtosis` for definitions of the robust kurtosis measures
    """
    pass

def robust_kurtosis(y, axis=0, ab=(5.0, 50.0), dg=(2.5, 25.0), excess=True):
    """
    Calculates the four kurtosis measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the kurtosis are computed.  If `None`, the
        entire array is used.
    a iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure
    excess : bool, optional
        If true (default), computed values are excess of those for a standard
        normal distribution.

    Returns
    -------
    kr1 : ndarray
          The standard kurtosis estimator.
    kr2 : ndarray
          Kurtosis estimator based on octiles.
    kr3 : ndarray
          Kurtosis estimators based on exceedance expectations.
    kr4 : ndarray
          Kurtosis measure based on the spread between high and low quantiles.

    Notes
    -----
    The robust kurtosis measures are defined

    .. math::

        KR_{2}=\\frac{\\left(\\hat{q}_{.875}-\\hat{q}_{.625}\\right)
        +\\left(\\hat{q}_{.375}-\\hat{q}_{.125}\\right)}
        {\\hat{q}_{.75}-\\hat{q}_{.25}}

    .. math::

        KR_{3}=\\frac{\\hat{E}\\left(y|y>\\hat{q}_{1-\\alpha}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\alpha}\\right)}
        {\\hat{E}\\left(y|y>\\hat{q}_{1-\\beta}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\beta}\\right)}

    .. math::

        KR_{4}=\\frac{\\hat{q}_{1-\\delta}-\\hat{q}_{\\delta}}
        {\\hat{q}_{1-\\gamma}-\\hat{q}_{\\gamma}}

    where :math:`\\hat{q}_{p}` is the estimated quantile at :math:`p`.

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    pass

def _medcouple_1d(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    pass

def medcouple(y, axis=0):
    """
    Calculate the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : {int, None}
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.

    Returns
    -------
    mc : ndarray
        The medcouple statistic with the same shape as `y`, with the specified
        axis removed.

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    pass