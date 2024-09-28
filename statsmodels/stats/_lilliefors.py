"""
Implements Lilliefors corrected Kolmogorov-Smirnov tests for normal and
exponential distributions.

`kstest_fit` is provided as a top-level function to access both tests.
`kstest_normal` and `kstest_exponential` are provided as convenience functions
with the appropriate test as the default.
`lilliefors` is provided as an alias for `kstest_fit`.

Created on Sat Oct 01 13:16:49 2011

Author: Josef Perktold
License: BSD-3

pvalues for Lilliefors test are based on formula and table in

An Analytic Approximation to the Distribution of Lilliefors's Test Statistic
for Normality
Author(s): Gerard E. Dallal and Leland WilkinsonSource: The American
Statistician, Vol. 40, No. 4 (Nov., 1986), pp. 294-296
Published by: American Statistical Association
Stable URL: http://www.jstor.org/stable/2684607 .

On the Kolmogorov-Smirnov Test for Normality with Mean and Variance Unknown
Hubert W. Lilliefors
Journal of the American Statistical Association, Vol. 62, No. 318.
(Jun., 1967), pp. 399-402.

---

Updated 2017-07-23
Jacob C. Kimmel

Ref:
Lilliefors, H.W.
On the Kolmogorov-Smirnov test for the exponential distribution with mean
unknown. Journal of the American Statistical Association, Vol 64, No. 325.
(1969), pp. 387–389.
"""
from functools import partial
import numpy as np
from scipy import stats
from statsmodels.tools.validation import string_like
from ._lilliefors_critical_values import critical_values, asymp_critical_values, PERCENTILES
from .tabledist import TableDist

def _make_asymptotic_function(params):
    """
    Generates an asymptotic distribution callable from a param matrix

    Polynomial is a[0] * x**(-1/2) + a[1] * x**(-1) + a[2] * x**(-3/2)

    Parameters
    ----------
    params : ndarray
        Array with shape (nalpha, 3) where nalpha is the number of
        significance levels
    """
    pass

def ksstat(x, cdf, alternative='two_sided', args=()):
    """
    Calculate statistic for the Kolmogorov-Smirnov test for goodness of fit

    This calculates the test statistic for a test of the distribution G(x) of
    an observed variable against a given distribution F(x). Under the null
    hypothesis the two distributions are identical, G(x)=F(x). The
    alternative hypothesis can be either 'two_sided' (default), 'less'
    or 'greater'. The KS test is only valid for continuous distributions.

    Parameters
    ----------
    x : array_like, 1d
        array of observations
    cdf : str or callable
        string: name of a distribution in scipy.stats
        callable: function to evaluate cdf
    alternative : 'two_sided' (default), 'less' or 'greater'
        defines the alternative hypothesis (see explanation)
    args : tuple, sequence
        distribution parameters for call to cdf


    Returns
    -------
    D : float
        KS test statistic, either D, D+ or D-

    See Also
    --------
    scipy.stats.kstest

    Notes
    -----

    In the one-sided test, the alternative is that the empirical
    cumulative distribution function of the random variable is "less"
    or "greater" than the cumulative distribution function F(x) of the
    hypothesis, G(x)<=F(x), resp. G(x)>=F(x).

    In contrast to scipy.stats.kstest, this function only calculates the
    statistic which can be used either as distance measure or to implement
    case specific p-values.
    """
    pass

def get_lilliefors_table(dist='norm'):
    """
    Generates tables for significance levels of Lilliefors test statistics

    Tables for available normal and exponential distribution testing,
    as specified in Lilliefors references above

    Parameters
    ----------
    dist : str
        distribution being tested in set {'norm', 'exp'}.

    Returns
    -------
    lf : TableDist object.
        table of critical values
    """
    pass
lilliefors_table_norm = get_lilliefors_table(dist='norm')
lilliefors_table_expon = get_lilliefors_table(dist='exp')

def pval_lf(d_max, n):
    """
    Approximate pvalues for Lilliefors test

    This is only valid for pvalues smaller than 0.1 which is not checked in
    this function.

    Parameters
    ----------
    d_max : array_like
        two-sided Kolmogorov-Smirnov test statistic
    n : int or float
        sample size

    Returns
    -------
    p-value : float or ndarray
        pvalue according to approximation formula of Dallal and Wilkinson.

    Notes
    -----
    This is mainly a helper function where the calling code should dispatch
    on bound violations. Therefore it does not check whether the pvalue is in
    the valid range.

    Precision for the pvalues is around 2 to 3 decimals. This approximation is
    also used by other statistical packages (e.g. R:fBasics) but might not be
    the most precise available.

    References
    ----------
    DallalWilkinson1986
    """
    pass

def kstest_fit(x, dist='norm', pvalmethod='table'):
    """
    Test assumed normal or exponential distribution using Lilliefors' test.

    Lilliefors' test is a Kolmogorov-Smirnov test with estimated parameters.

    Parameters
    ----------
    x : array_like, 1d
        Data to test.
    dist : {'norm', 'exp'}, optional
        The assumed distribution.
    pvalmethod : {'approx', 'table'}, optional
        The method used to compute the p-value of the test statistic. In
        general, 'table' is preferred and makes use of a very large simulation.
        'approx' is only valid for normality. if `dist = 'exp'` `table` is
        always used. 'approx' uses the approximation formula of Dalal and
        Wilkinson, valid for pvalues < 0.1. If the pvalue is larger than 0.1,
        then the result of `table` is returned.

    Returns
    -------
    ksstat : float
        Kolmogorov-Smirnov test statistic with estimated mean and variance.
    pvalue : float
        If the pvalue is lower than some threshold, e.g. 0.05, then we can
        reject the Null hypothesis that the sample comes from a normal
        distribution.

    Notes
    -----
    'table' uses an improved table based on 10,000,000 simulations. The
    critical values are approximated using
    log(cv_alpha) = b_alpha + c[0] log(n) + c[1] log(n)**2
    where cv_alpha is the critical value for a test with size alpha,
    b_alpha is an alpha-specific intercept term and c[1] and c[2] are
    coefficients that are shared all alphas.
    Values in the table are linearly interpolated. Values outside the
    range are be returned as bounds, 0.990 for large and 0.001 for small
    pvalues.

    For implementation details, see  lilliefors_critical_value_simulation.py in
    the test directory.
    """
    pass
lilliefors = kstest_fit
kstest_normal = kstest_fit
kstest_exponential = partial(kstest_fit, dist='exp')