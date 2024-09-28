"""extra statistical function and helper functions

contains:

* goodness-of-fit tests
  - powerdiscrepancy
  - gof_chisquare_discrete
  - gof_binning_discrete



Author: Josef Perktold
License : BSD-3

changes
-------
2013-02-25 : add chisquare_power, effectsize and "value"

"""
from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats

def powerdiscrepancy(observed, expected, lambd=0.0, axis=0, ddof=0):
    """Calculates power discrepancy, a class of goodness-of-fit tests
    as a measure of discrepancy between observed and expected data.

    This contains several goodness-of-fit tests as special cases, see the
    description of lambd, the exponent of the power discrepancy. The pvalue
    is based on the asymptotic chi-square distribution of the test statistic.

    freeman_tukey:
    D(x|\\theta) = \\sum_j (\\sqrt{x_j} - \\sqrt{e_j})^2

    Parameters
    ----------
    o : Iterable
        Observed values
    e : Iterable
        Expected values
    lambd : {float, str}
        * float : exponent `a` for power discrepancy
        * 'loglikeratio': a = 0
        * 'freeman_tukey': a = -0.5
        * 'pearson': a = 1   (standard chisquare test statistic)
        * 'modified_loglikeratio': a = -1
        * 'cressie_read': a = 2/3
        * 'neyman' : a = -2 (Neyman-modified chisquare, reference from a book?)
    axis : int
        axis for observations of one series
    ddof : int
        degrees of freedom correction,

    Returns
    -------
    D_obs : Discrepancy of observed values
    pvalue : pvalue


    References
    ----------
    Cressie, Noel  and Timothy R. C. Read, Multinomial Goodness-of-Fit Tests,
        Journal of the Royal Statistical Society. Series B (Methodological),
        Vol. 46, No. 3 (1984), pp. 440-464

    Campbell B. Read: Freeman-Tukey chi-squared goodness-of-fit statistics,
        Statistics & Probability Letters 18 (1993) 271-278

    Nobuhiro Taneichi, Yuri Sekiya, Akio Suzukawa, Asymptotic Approximations
        for the Distributions of the Multinomial Goodness-of-Fit Statistics
        under Local Alternatives, Journal of Multivariate Analysis 81, 335?359 (2002)
    Steele, M. 1,2, C. Hurst 3 and J. Chaseling, Simulated Power of Discrete
        Goodness-of-Fit Tests for Likert Type Data

    Examples
    --------

    >>> observed = np.array([ 2.,  4.,  2.,  1.,  1.])
    >>> expected = np.array([ 0.2,  0.2,  0.2,  0.2,  0.2])

    for checking correct dimension with multiple series

    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd='freeman_tukey',axis=1)
    (array([[ 2.745166,  2.745166]]), array([[ 0.6013346,  0.6013346]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=0,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=1,axis=1)
    (array([[ 3.,  3.]]), array([[ 0.5578254,  0.5578254]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))

    each random variable can have different total count/sum

    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), 20*expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=-1, axis=0)
    (array([[ 2.77258872,  5.54517744]]), array([[ 0.59657359,  0.2357868 ]]))
    """
    pass

def gof_chisquare_discrete(distfn, arg, rvs, alpha, msg):
    """perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : str
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    Notes
    -----
    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    """
    pass

def gof_binning_discrete(rvs, distfn, arg, nsupp=20):
    """get bins for chisquare type gof tests for a discrete distribution

    Parameters
    ----------
    rvs : ndarray
        sample data
    distname : str
        name of distribution function
    arg : sequence
        parameters of distribution
    nsupp : int
        number of bins. The algorithm tries to find bins with equal weights.
        depending on the distribution, the actual number of bins can be smaller.

    Returns
    -------
    freq : ndarray
        empirical frequencies for sample; not normalized, adds up to sample size
    expfreq : ndarray
        theoretical frequencies according to distribution
    histsupp : ndarray
        bin boundaries for histogram, (added 1e-8 for numerical robustness)

    Notes
    -----
    The results can be used for a chisquare test ::

        (chis,pval) = stats.chisquare(freq, expfreq)

    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    todo :
      optimal number of bins ? (check easyfit),
      recommendation in literature at least 5 expected observations in each bin

    """
    pass
'Extension to chisquare goodness-of-fit test\n\nCreated on Mon Feb 25 13:46:53 2013\n\nAuthor: Josef Perktold\nLicense: BSD-3\n'

def chisquare(f_obs, f_exp=None, value=0, ddof=0, return_basic=True):
    """chisquare goodness-of-fit test

    The null hypothesis is that the distance between the expected distribution
    and the observed frequencies is ``value``. The alternative hypothesis is
    that the distance is larger than ``value``. ``value`` is normalized in
    terms of effect size.

    The standard chisquare test has the null hypothesis that ``value=0``, that
    is the distributions are the same.


    Notes
    -----
    The case with value greater than zero is similar to an equivalence test,
    that the exact null hypothesis is replaced by an approximate hypothesis.
    However, TOST "reverses" null and alternative hypothesis, while here the
    alternative hypothesis is that the distance (divergence) is larger than a
    threshold.

    References
    ----------
    McLaren, ...
    Drost,...

    See Also
    --------
    powerdiscrepancy
    scipy.stats.chisquare

    """
    pass

def chisquare_power(effect_size, nobs, n_bins, alpha=0.05, ddof=0):
    """power of chisquare goodness of fit test

    effect size is sqrt of chisquare statistic divided by nobs

    Parameters
    ----------
    effect_size : float
        This is the deviation from the Null of the normalized chi_square
        statistic. This follows Cohen's definition (sqrt).
    nobs : int or float
        number of observations
    n_bins : int (or float)
        number of bins, or points in the discrete distribution
    alpha : float in (0,1)
        significance level of the test, default alpha=0.05

    Returns
    -------
    power : float
        power of the test at given significance level at effect size

    Notes
    -----
    This function also works vectorized if all arguments broadcast.

    This can also be used to calculate the power for power divergence test.
    However, for the range of more extreme values of the power divergence
    parameter, this power is not a very good approximation for samples of
    small to medium size (Drost et al. 1989)

    References
    ----------
    Drost, ...

    See Also
    --------
    chisquare_effectsize
    statsmodels.stats.GofChisquarePower

    """
    pass

def chisquare_effectsize(probs0, probs1, correction=None, cohen=True, axis=0):
    """effect size for a chisquare goodness-of-fit test

    Parameters
    ----------
    probs0 : array_like
        probabilities or cell frequencies under the Null hypothesis
    probs1 : array_like
        probabilities or cell frequencies under the Alternative hypothesis
        probs0 and probs1 need to have the same length in the ``axis`` dimension.
        and broadcast in the other dimensions
        Both probs0 and probs1 are normalized to add to one (in the ``axis``
        dimension).
    correction : None or tuple
        If None, then the effect size is the chisquare statistic divide by
        the number of observations.
        If the correction is a tuple (nobs, df), then the effectsize is
        corrected to have less bias and a smaller variance. However, the
        correction can make the effectsize negative. In that case, the
        effectsize is set to zero.
        Pederson and Johnson (1990) as referenced in McLaren et all. (1994)
    cohen : bool
        If True, then the square root is returned as in the definition of the
        effect size by Cohen (1977), If False, then the original effect size
        is returned.
    axis : int
        If the probability arrays broadcast to more than 1 dimension, then
        this is the axis over which the sums are taken.

    Returns
    -------
    effectsize : float
        effect size of chisquare test

    """
    pass