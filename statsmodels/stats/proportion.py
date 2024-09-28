"""
Tests and Confidence Intervals for Binomial Proportions

Created on Fri Mar 01 00:23:07 2013

Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lzip
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
FLOAT_INFO = np.finfo(float)

def _bound_proportion_confint(func: Callable[[float], float], qi: float, lower: bool=True) -> float:
    """
    Try hard to find a bound different from eps/1 - eps in proportion_confint

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    qi : float
        The empirical success rate
    lower : bool
        Whether to fund a lower bound for the left side of the CI

    Returns
    -------
    float
        The coarse bound
    """
    pass

def _bisection_search_conservative(func: Callable[[float], float], lb: float, ub: float, steps: int=27) -> Tuple[float, float]:
    """
    Private function used as a fallback by proportion_confint

    Used when brentq returns a non-conservative bound for the CI

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    lb : float
        Lower bound
    ub : float
        Upper bound
    steps : int
        Number of steps to use in the bisection

    Returns
    -------
    est : float
        The estimated value.  Will always produce a negative value of func
    func_val : float
        The value of the function at the estimate
    """
    pass

def proportion_confint(count, nobs, alpha: float=0.05, method='normal'):
    """
    Confidence interval for a binomial proportion

    Parameters
    ----------
    count : {int or float, array_like}
        number of successes, can be pandas Series or DataFrame. Arrays
        must contain integer values if method is "binom_test".
    nobs : {int or float, array_like}
        total number of trials.  Arrays must contain integer values if method
        is "binom_test".
    alpha : float
        Significance level, default 0.05. Must be in (0, 1)
    method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}
        default: "normal"
        method to use for confidence interval. Supported methods:

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : Numerical inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : {float, ndarray, Series DataFrame}
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The "beta" and "jeffreys" interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of "beta".

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of "normal" and "agresti_coull".

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    .. [*] https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    .. [*] Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001).
       "Interval Estimation for a Binomial Proportion", Statistical
       Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
    """
    pass

def multinomial_proportions_confint(counts, alpha=0.05, method='goodman'):
    """
    Confidence intervals for multinomial proportions.

    Parameters
    ----------
    counts : array_like of int, 1-D
        Number of observations in each category.
    alpha : float in (0, 1), optional
        Significance level, defaults to 0.05.
    method : {'goodman', 'sison-glaz'}, optional
        Method to use to compute the confidence intervals; available methods
        are:

         - `goodman`: based on a chi-squared approximation, valid if all
           values in `counts` are greater or equal to 5 [2]_
         - `sison-glaz`: less conservative than `goodman`, but only valid if
           `counts` has 7 or more categories (``len(counts) >= 7``) [3]_

    Returns
    -------
    confint : ndarray, 2-D
        Array of [lower, upper] confidence levels for each category, such that
        overall coverage is (approximately) `1-alpha`.

    Raises
    ------
    ValueError
        If `alpha` is not in `(0, 1)` (bounds excluded), or if the values in
        `counts` are not all positive or null.
    NotImplementedError
        If `method` is not kown.
    Exception
        When ``method == 'sison-glaz'``, if for some reason `c` cannot be
        computed; this signals a bug and should be reported.

    Notes
    -----
    The `goodman` method [2]_ is based on approximating a statistic based on
    the multinomial as a chi-squared random variable. The usual recommendation
    is that this is valid if all the values in `counts` are greater than or
    equal to 5. There is no condition on the number of categories for this
    method.

    The `sison-glaz` method [3]_ approximates the multinomial probabilities,
    and evaluates that with a maximum-likelihood estimator. The first
    approximation is an Edgeworth expansion that converges when the number of
    categories goes to infinity, and the maximum-likelihood estimator converges
    when the number of observations (``sum(counts)``) goes to infinity. In
    their paper, Sison & Glaz demo their method with at least 7 categories, so
    ``len(counts) >= 7`` with all values in `counts` at or above 5 can be used
    as a rule of thumb for the validity of this method. This method is less
    conservative than the `goodman` method (i.e. it will yield confidence
    intervals closer to the desired significance level), but produces
    confidence intervals of uniform width over all categories (except when the
    intervals reach 0 or 1, in which case they are truncated), which makes it
    most useful when proportions are of similar magnitude.

    Aside from the original sources ([1]_, [2]_, and [3]_), the implementation
    uses the formulas (though not the code) presented in [4]_ and [5]_.

    References
    ----------
    .. [1] Levin, Bruce, "A representation for multinomial cumulative
           distribution functions," The Annals of Statistics, Vol. 9, No. 5,
           1981, pp. 1123-1126.

    .. [2] Goodman, L.A., "On simultaneous confidence intervals for multinomial
           proportions," Technometrics, Vol. 7, No. 2, 1965, pp. 247-254.

    .. [3] Sison, Cristina P., and Joseph Glaz, "Simultaneous Confidence
           Intervals and Sample Size Determination for Multinomial
           Proportions," Journal of the American Statistical Association,
           Vol. 90, No. 429, 1995, pp. 366-369.

    .. [4] May, Warren L., and William D. Johnson, "A SAS® macro for
           constructing simultaneous confidence intervals  for multinomial
           proportions," Computer methods and programs in Biomedicine, Vol. 53,
           No. 3, 1997, pp. 153-162.

    .. [5] May, Warren L., and William D. Johnson, "Constructing two-sided
           simultaneous confidence intervals for multinomial proportions for
           small counts in a large number of cells," Journal of Statistical
           Software, Vol. 5, No. 6, 2000, pp. 1-24.
    """
    pass

def samplesize_confint_proportion(proportion, half_length, alpha=0.05, method='normal'):
    """
    Find sample size to get desired confidence interval length

    Parameters
    ----------
    proportion : float in (0, 1)
        proportion or quantile
    half_length : float in (0, 1)
        desired half length of the confidence interval
    alpha : float in (0, 1)
        significance level, default 0.05,
        coverage of the two-sided interval is (approximately) ``1 - alpha``
    method : str in ['normal']
        method to use for confidence interval,
        currently only normal approximation

    Returns
    -------
    n : float
        sample size to get the desired half length of the confidence interval

    Notes
    -----
    this is mainly to store the formula.
    possible application: number of replications in bootstrap samples

    """
    pass

def proportion_effectsize(prop1, prop2, method='normal'):
    """
    Effect size for a test comparing two proportions

    for use in power function

    Parameters
    ----------
    prop1, prop2 : float or array_like
        The proportion value(s).

    Returns
    -------
    es : float or ndarray
        effect size for (transformed) prop1 - prop2

    Notes
    -----
    only method='normal' is implemented to match pwr.p2.test
    see http://www.statmethods.net/stats/power.html

    Effect size for `normal` is defined as ::

        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

    I think other conversions to normality can be used, but I need to check.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> sm.stats.proportion_effectsize(0.5, 0.4)
    0.20135792079033088
    >>> sm.stats.proportion_effectsize([0.3, 0.4, 0.5], 0.4)
    array([-0.21015893,  0.        ,  0.20135792])

    """
    pass

def std_prop(prop, nobs):
    """
    Standard error for the estimate of a proportion

    This is just ``np.sqrt(p * (1. - p) / nobs)``

    Parameters
    ----------
    prop : array_like
        proportion
    nobs : int, array_like
        number of observations

    Returns
    -------
    std : array_like
        standard error for a proportion of nobs independent observations
    """
    pass

def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt, alpha=0.05, discrete=True, dist='norm', nobs=None, continuity=0, critval_continuity=0):
    """
    Generic statistical power function for normal based equivalence test

    This includes options to adjust the normal approximation and can use
    the binomial to evaluate the probability of the rejection region

    see power_ztost_prob for a description of the options
    """
    pass

def binom_tost(count, nobs, low, upp):
    """
    Exact TOST test for one proportion using binomial distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    low, upp : floats
        lower and upper limit of equivalence region

    Returns
    -------
    pvalue : float
        p-value of equivalence test
    pval_low, pval_upp : floats
        p-values of lower and upper one-sided tests

    """
    pass

def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    """
    Rejection region for binomial TOST

    The interval includes the end points,
    `reject` if and only if `r_low <= x <= r_upp`.

    The interval might be empty with `r_upp < r_low`.

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : float
        lower and upper bound of rejection region

    """
    pass

def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='two-sided'):
    """
    Rejection region for binomial test for one sample proportion

    The interval includes the end points of the rejection region.

    Parameters
    ----------
    value : float
        proportion under the Null hypothesis
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : int
        lower and upper bound of rejection region
    """
    pass

def binom_test(count, nobs, prop=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    prop : float, optional
        The probability of success under the null hypothesis,
        `0 <= prop <= 1`. The default value is `prop = 0.5`
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test

    Notes
    -----
    This uses scipy.stats.binom_test for the two-sided alternative.
    """
    pass

def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm', variance_prop=None, discrete=True, continuity=0, critval_continuity=0):
    """
    Power of proportions equivalence test based on normal distribution

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        number of observations
    p_alt : float in (0,1)
        proportion under the alternative
    alpha : float in (0,1)
        significance level of the test
    dist : str in ['norm', 'binom']
        This defines the distribution to evaluate the power of the test. The
        critical values of the TOST test are always based on the normal
        approximation, but the distribution for the power can be either the
        normal (default) or the binomial (exact) distribution.
    variance_prop : None or float in (0,1)
        If this is None, then the variances for the two one sided tests are
        based on the proportions equal to the equivalence limits.
        If variance_prop is given, then it is used to calculate the variance
        for the TOST statistics. If this is based on an sample, then the
        estimated proportion can be used.
    discrete : bool
        If true, then the critical values of the rejection region are converted
        to integers. If dist is "binom", this is automatically assumed.
        If discrete is false, then the TOST critical values are used as
        floating point numbers, and the power is calculated based on the
        rejection region that is not discretized.
    continuity : bool or float
        adjust the rejection region for the normal power probability. This has
        and effect only if ``dist='norm'``
    critval_continuity : bool or float
        If this is non-zero, then the critical values of the tost rejection
        region are adjusted before converting to integers. This affects both
        distributions, ``dist='norm'`` and ``dist='binom'``.

    Returns
    -------
    power : float
        statistical power of the equivalence test.
    (k_low, k_upp, z_low, z_upp) : tuple of floats
        critical limits in intermediate steps
        temporary return, will be changed

    Notes
    -----
    In small samples the power for the ``discrete`` version, has a sawtooth
    pattern as a function of the number of observations. As a consequence,
    small changes in the number of observations or in the normal approximation
    can have a large effect on the power.

    ``continuity`` and ``critval_continuity`` are added to match some results
    of PASS, and are mainly to investigate the sensitivity of the ztost power
    to small changes in the rejection region. From my interpretation of the
    equations in the SAS manual, both are zero in SAS.

    works vectorized

    **verification:**

    The ``dist='binom'`` results match PASS,
    The ``dist='norm'`` results look reasonable, but no benchmark is available.

    References
    ----------
    SAS Manual: Chapter 68: The Power Procedure, Computational Resources
    PASS Chapter 110: Equivalence Tests for One Proportion.

    """
    pass

def _table_proportion(count, nobs):
    """
    Create a k by 2 contingency table for proportion

    helper function for proportions_chisquare

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.

    Returns
    -------
    table : ndarray
        (k, 2) contingency table

    Notes
    -----
    recent scipy has more elaborate contingency table functions

    """
    pass

def proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False):
    """
    Test for proportions based on normal (z) test

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : {int, array_like}
        the number of trials or observations, with the same length as
        count.
    value : float, array_like or None, optional
        This is the value of the null hypothesis equal to the proportion in the
        case of a one sample test. In the case of a two-sample test, the
        null hypothesis is that prop[0] - prop[1] = value, where prop is the
        proportion in the two samples. If not provided value = 0 and the null
        is prop[0] = prop[1]
    alternative : str in ['two-sided', 'smaller', 'larger']
        The alternative hypothesis can be either two-sided or one of the one-
        sided tests, smaller means that the alternative hypothesis is
        ``prop < value`` and larger means ``prop > value``. In the two sample
        test, smaller means that the alternative hypothesis is ``p1 < p2`` and
        larger means ``p1 > p2`` where ``p1`` is the proportion of the first
        sample and ``p2`` of the second one.
    prop_var : False or float in (0, 1)
        If prop_var is false, then the variance of the proportion estimate is
        calculated based on the sample proportion. Alternatively, a proportion
        can be specified to calculate this variance. Common use case is to
        use the proportion under the Null hypothesis to specify the variance
        of the proportion estimate.

    Returns
    -------
    zstat : float
        test statistic for the z-test
    p-value : float
        p-value for the z-test

    Examples
    --------
    >>> count = 5
    >>> nobs = 83
    >>> value = .05
    >>> stat, pval = proportions_ztest(count, nobs, value)
    >>> print('{0:0.3f}'.format(pval))
    0.695

    >>> import numpy as np
    >>> from statsmodels.stats.proportion import proportions_ztest
    >>> count = np.array([5, 12])
    >>> nobs = np.array([83, 99])
    >>> stat, pval = proportions_ztest(count, nobs)
    >>> print('{0:0.3f}'.format(pval))
    0.159

    Notes
    -----
    This uses a simple normal test for proportions. It should be the same as
    running the mean z-test on the data encoded 1 for event and 0 for no event
    so that the sum corresponds to the count.

    In the one and two sample cases with two-sided alternative, this test
    produces the same p-value as ``proportions_chisquare``, since the
    chisquare is the distribution of the square of a standard normal
    distribution.
    """
    pass

def proportions_ztost(count, nobs, low, upp, prop_var='sample'):
    """
    Equivalence test based on normal distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : int
        the number of trials or observations, with the same length as
        count.
    low, upp : float
        equivalence interval low < prop1 - prop2 < upp
    prop_var : str or float in (0, 1)
        prop_var determines which proportion is used for the calculation
        of the standard deviation of the proportion estimate
        The available options for string are 'sample' (default), 'null' and
        'limits'. If prop_var is a float, then it is used directly.

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    checked only for 1 sample case

    """
    pass

def proportions_chisquare(count, nobs, value=None):
    """
    Test for proportions based on chisquare test

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : int
        the number of trials or observations, with the same length as
        count.
    value : None or float or array_like

    Returns
    -------
    chi2stat : float
        test statistic for the chisquare test
    p-value : float
        p-value for the chisquare test
    (table, expected)
        table is a (k, 2) contingency table, ``expected`` is the corresponding
        table of counts that are expected under independence with given
        margins

    Notes
    -----
    Recent version of scipy.stats have a chisquare test for independence in
    contingency tables.

    This function provides a similar interface to chisquare tests as
    ``prop.test`` in R, however without the option for Yates continuity
    correction.

    count can be the count for the number of events for a single proportion,
    or the counts for several independent proportions. If value is given, then
    all proportions are jointly tested against this value. If value is not
    given and count and nobs are not scalar, then the null hypothesis is
    that all samples have the same proportion.

    """
    pass

def proportions_chisquare_allpairs(count, nobs, multitest_method='hs'):
    """
    Chisquare test of proportions for all pairs of k samples

    Performs a chisquare test for proportions for all pairwise comparisons.
    The alternative is two-sided

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.

    Notes
    -----
    Yates continuity correction is not available.
    """
    pass

def proportions_chisquare_pairscontrol(count, nobs, value=None, multitest_method='hs', alternative='two-sided'):
    """
    Chisquare test of proportions for pairs of k samples compared to control

    Performs a chisquare test for proportions for pairwise comparisons with a
    control (Dunnet's test). The control is assumed to be the first element
    of ``count`` and ``nobs``. The alternative is two-sided, larger or
    smaller.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.


    Notes
    -----
    Yates continuity correction is not available.

    ``value`` and ``alternative`` options are not yet implemented.

    """
    pass

def confint_proportions_2indep(count1, nobs1, count2, nobs2, method=None, compare='diff', alpha=0.05, correction=True):
    """
    Confidence intervals for comparing two independent proportions.

    This assumes that we have two independent binomial samples.

    Parameters
    ----------
    count1, nobs1 : float
        Count and sample size for first sample.
    count2, nobs2 : float
        Count and sample size for the second sample.
    method : str
        Method for computing confidence interval. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:
         - 'wald',
         - 'agresti-caffo'
         - 'newcomb' (default)
         - 'score'

        ratio:
         - 'log'
         - 'log-adjusted' (default)
         - 'score'

        odds-ratio:
         - 'logit'
         - 'logit-adjusted' (default)
         - 'score'

    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is diff, then the confidence interval is for diff = p1 - p2.
        If compare is ratio, then the confidence interval is for the risk ratio
        defined by ratio = p1 / p2.
        If compare is odds-ratio, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    alpha : float
        Significance level for the confidence interval, default is 0.05.
        The nominal coverage probability is 1 - alpha.

    Returns
    -------
    low, upp

    See Also
    --------
    test_proportions_2indep
    tost_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.
        more ``methods`` will be added.

    References
    ----------
    .. [1] Fagerland, Morten W., Stian Lydersen, and Petter Laake. 2015.
       “Recommended Confidence Intervals for Two Independent Binomial
       Proportions.” Statistical Methods in Medical Research 24 (2): 224–54.
       https://doi.org/10.1177/0962280211415469.
    .. [2] Koopman, P. A. R. 1984. “Confidence Intervals for the Ratio of Two
       Binomial Proportions.” Biometrics 40 (2): 513–17.
       https://doi.org/10.2307/2531405.
    .. [3] Miettinen, Olli, and Markku Nurminen. "Comparative analysis of two
       rates." Statistics in medicine 4, no. 2 (1985): 213-226.
    .. [4] Newcombe, Robert G. 1998. “Interval Estimation for the Difference
       between Independent Proportions: Comparison of Eleven Methods.”
       Statistics in Medicine 17 (8): 873–90.
       https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<873::AID-
       SIM779>3.0.CO;2-I.
    .. [5] Newcombe, Robert G., and Markku M. Nurminen. 2011. “In Defence of
       Score Intervals for Proportions and Their Differences.” Communications
       in Statistics - Theory and Methods 40 (7): 1271–82.
       https://doi.org/10.1080/03610920903576580.
    """
    pass

def _shrink_prob(count1, nobs1, count2, nobs2, shrink_factor=2, return_corr=True):
    """
    Shrink observed counts towards independence

    Helper function for 'logit-smoothed' inference for the odds-ratio of two
    independent proportions.

    Parameters
    ----------
    count1, nobs1 : float or int
        count and sample size for first sample
    count2, nobs2 : float or int
        count and sample size for the second sample
    shrink_factor : float
        This corresponds to the number of observations that are added in total
        proportional to the probabilities under independence.
    return_corr : bool
        If true, then only the correction term is returned
        If false, then the corrected counts, i.e. original counts plus
        correction term, are returned.

    Returns
    -------
    count1_corr, nobs1_corr, count2_corr, nobs2_corr : float
        correction or corrected counts
    prob_indep :
        TODO/Warning : this will change most likely
        probabilities under independence, only returned if return_corr is
        false.

    """
    pass

def score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    """
    Score test for two independent proportions

    This uses the constrained estimate of the proportions to compute
    the variance under the Null hypothesis.

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    value : float
        diff, ratio or odds-ratio under the null hypothesis. If value is None,
        then equality of proportions under the Null is assumed,
        i.e. value=0 for 'diff' or value=1 for either rate or odds-ratio.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is diff, then the confidence interval is for diff = p1 - p2.
        If compare is ratio, then the confidence interval is for the risk ratio
        defined by ratio = p1 / p2.
        If compare is odds-ratio, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    Notes
    -----
    Status: experimental, the type or extra information in the return might
    change.

    """
    pass

def test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, method=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    """
    Hypothesis test for comparing two independent proportions

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis are

    for compare = 'diff'

    - H0: prop1 - prop2 - value = 0
    - H1: prop1 - prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 - prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 - prop2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: prop1 / prop2 - value = 0
    - H1: prop1 / prop2 - value != 0  if alternative = 'two-sided'
    - H1: prop1 / prop2 - value > 0   if alternative = 'larger'
    - H1: prop1 / prop2 - value < 0   if alternative = 'smaller'

    for compare = 'odds-ratio'

    - H0: or - value = 0
    - H1: or - value != 0  if alternative = 'two-sided'
    - H1: or - value > 0   if alternative = 'larger'
    - H1: or - value < 0   if alternative = 'smaller'

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1 : int
        Count for first sample.
    nobs1 : int
        Sample size for first sample.
    count2 : int
        Count for the second sample.
    nobs2 : int
        Sample size for the second sample.
    value : float
        Value of the difference, risk ratio or odds ratio of 2 independent
        proportions under the null hypothesis.
        Default is equal proportions, 0 for diff and 1 for risk-ratio and for
        odds-ratio.
    method : string
        Method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:

        - 'wald',
        - 'agresti-caffo'
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        ratio:

        - 'log': wald test using log transformation
        - 'log-adjusted': wald test using log transformation,
           adds 0.5 to counts
        - 'score': if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

        odds-ratio:

        - 'logit': wald test using logit transformation
        - 'logit-adjusted': wald test using logit transformation,
           adds 0.5 to counts
        - 'logit-smoothed': wald test using logit transformation, biases
           cell counts towards independence by adding two observations in
           total.
        - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : {'diff', 'ratio' 'odds-ratio'}
        If compare is `diff`, then the hypothesis test is for the risk
        difference diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    alternative : {'two-sided', 'smaller', 'larger'}
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    See Also
    --------
    tost_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.
        More ``methods`` will be added.

    The current default methods are

    - 'diff': 'agresti-caffo',
    - 'ratio': 'log-adjusted',
    - 'odds-ratio': 'logit-adjusted'

    """
    pass

def tost_proportions_2indep(count1, nobs1, count2, nobs2, low, upp, method=None, compare='diff', correction=True):
    """
    Equivalence test based on two one-sided `test_proportions_2indep`

    This assumes that we have two independent binomial samples.

    The Null and alternative hypothesis for equivalence testing are

    for compare = 'diff'

    - H0: prop1 - prop2 <= low or upp <= prop1 - prop2
    - H1: low < prop1 - prop2 < upp

    for compare = 'ratio'

    - H0: prop1 / prop2 <= low or upp <= prop1 / prop2
    - H1: low < prop1 / prop2 < upp


    for compare = 'odds-ratio'

    - H0: or <= low or upp <= or
    - H1: low < or < upp

    where odds-ratio or = prop1 / (1 - prop1) / (prop2 / (1 - prop2))

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    low, upp :
        equivalence margin for diff, risk ratio or odds ratio
    method : string
        method for computing the hypothesis test. If method is None, then a
        default method is used. The default might change as more methods are
        added.

        diff:
         - 'wald',
         - 'agresti-caffo'
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        ratio:
         - 'log': wald test using log transformation
         - 'log-adjusted': wald test using log transformation,
            adds 0.5 to counts
         - 'score' if correction is True, then this uses the degrees of freedom
           correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985.

        odds-ratio:
         - 'logit': wald test using logit transformation
         - 'logit-adjusted': : wald test using logit transformation,
            adds 0.5 to counts
         - 'logit-smoothed': : wald test using logit transformation, biases
            cell counts towards independence by adding two observations in
            total.
         - 'score' if correction is True, then this uses the degrees of freedom
            correction ``nobs / (nobs - 1)`` as in Miettinen Nurminen 1985

    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the hypothesis test is for
        diff = p1 - p2.
        If compare is `ratio`, then the hypothesis test is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the hypothesis test is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    pvalue : float
        p-value is the max of the pvalues of the two one-sided tests
    t1 : test results
        results instance for one-sided hypothesis at the lower margin
    t1 : test results
        results instance for one-sided hypothesis at the upper margin

    See Also
    --------
    test_proportions_2indep
    confint_proportions_2indep

    Notes
    -----
    Status: experimental, API and defaults might still change.

    The TOST equivalence test delegates to `test_proportions_2indep` and has
    the same method and comparison options.

    """
    pass

def _std_2prop_power(diff, p2, ratio=1, alpha=0.05, value=0):
    """
    Compute standard error under null and alternative for 2 proportions

    helper function for power and sample size computation

    """
    pass

def power_proportions_2indep(diff, prop2, nobs1, ratio=1, alpha=0.05, value=0, alternative='two-sided', return_results=True):
    """
    Power for ztest that two independent proportions are equal

    This assumes that the variance is based on the pooled proportion
    under the null and the non-pooled variance under the alternative

    Parameters
    ----------
    diff : float
        difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computed using p2 and diff
        p1 = p2 + diff
    nobs1 : float or int
        number of observations in sample 1
    ratio : float
        sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only the power is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        p_pooled
            pooled proportion, used for std_null
        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))
    """
    pass

def samplesize_proportions_2indep_onetail(diff, prop2, power, ratio=1, alpha=0.05, value=0, alternative='two-sided'):
    """
    Required sample size assuming normal distribution based on one tail

    This uses an explicit computation for the sample size that is required
    to achieve a given power corresponding to the appropriate tails of the
    normal distribution. This ignores the far tail in a two-sided test
    which is negligible in the common case when alternative and null are
    far apart.

    Parameters
    ----------
    diff : float
        Difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computing using p2 and diff
        p1 = p2 + diff
    power : float
        Power for which sample size is computed.
    ratio : float
        Sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        Currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. In the case of a one-sided
        alternative, it is assumed that the test is in the appropriate tail.

    Returns
    -------
    nobs1 : float
        Number of observations in sample 1.
    """
    pass

def _score_confint_inversion(count1, nobs1, count2, nobs2, compare='diff', alpha=0.05, correction=True):
    """
    Compute score confidence interval by inverting score test

    Parameters
    ----------
    count1, nobs1 :
        Count and sample size for first sample.
    count2, nobs2 :
        Count and sample size for the second sample.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is `diff`, then the confidence interval is for
        diff = p1 - p2.
        If compare is `ratio`, then the confidence interval is for the
        risk ratio defined by ratio = p1 / p2.
        If compare is `odds-ratio`, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2).
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    correction : bool
        If correction is True (default), then the Miettinen and Nurminen
        small sample correction to the variance nobs / (nobs - 1) is used.
        Applies only if method='score'.

    Returns
    -------
    low : float
        Lower confidence bound.
    upp : float
        Upper confidence bound.
    """
    pass

def _confint_riskratio_koopman(count1, nobs1, count2, nobs2, alpha=0.05, correction=True):
    """
    Score confidence interval for ratio or proportions, Koopman/Nam

    signature not consistent with other functions

    When correction is True, then the small sample correction nobs / (nobs - 1)
    by Miettinen/Nurminen is used.
    """
    pass

def _confint_riskratio_paired_nam(table, alpha=0.05):
    """
    Confidence interval for marginal risk ratio for matched pairs

    need full table

             success fail  marginal
    success    x11    x10  x1.
    fail       x01    x00  x0.
    marginal   x.1    x.0   n

    The confidence interval is for the ratio p1 / p0 where
    p1 = x1. / n and
    p0 - x.1 / n
    Todo: rename p1 to pa and p2 to pb, so we have a, b for treatment and
    0, 1 for success/failure

    current namings follow Nam 2009

    status
    testing:
    compared to example in Nam 2009
    internal polynomial coefficients in calculation correspond at around
        4 decimals
    confidence interval agrees only at 2 decimals

    """
    pass