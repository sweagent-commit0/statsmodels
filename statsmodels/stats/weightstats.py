"""Tests and descriptive statistics with weights


Created on 2010-09-18

Author: josef-pktd
License: BSD (3-clause)


References
----------
SPSS manual
SAS manual

This follows in large parts the SPSS manual, which is largely the same as
the SAS manual with different, simpler notation.

Freq, Weight in SAS seems redundant since they always show up as product, SPSS
has only weights.

Notes
-----

This has potential problems with ddof, I started to follow numpy with ddof=0
by default and users can change it, but this might still mess up the t-tests,
since the estimates for the standard deviation will be based on the ddof that
the user chooses.
- fixed ddof for the meandiff ttest, now matches scipy.stats.ttest_ind

Note: scipy has now a separate, pooled variance option in ttest, but I have not
compared yet.

"""
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly

class DescrStatsW:
    """
    Descriptive statistics and tests with weights for case weights

    Assumes that the data is 1d or 2d with (nobs, nvars) observations in rows,
    variables in columns, and that the same weight applies to each column.

    If degrees of freedom correction is used, then weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.

    This is essentially the same as replicating each observations by its
    weight, if the weights are integers, often called case or frequency weights.

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        dataset
    weights : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    ddof : int
        default ddof=0, degrees of freedom correction used for second moments,
        var, std, cov, corrcoef.
        However, statistical tests are independent of `ddof`, based on the
        standard formulas.

    Examples
    --------

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x1_2d = 1.0 + np.random.randn(20, 3)
    >>> w1 = np.random.randint(1, 4, 20)
    >>> d1 = DescrStatsW(x1_2d, weights=w1)
    >>> d1.mean
    array([ 1.42739844,  1.23174284,  1.083753  ])
    >>> d1.var
    array([ 0.94855633,  0.52074626,  1.12309325])
    >>> d1.std_mean
    array([ 0.14682676,  0.10878944,  0.15976497])

    >>> tstat, pval, df = d1.ttest_mean(0)
    >>> tstat; pval; df
    array([  9.72165021,  11.32226471,   6.78342055])
    array([  1.58414212e-12,   1.26536887e-14,   2.37623126e-08])
    44.0

    >>> tstat, pval, df = d1.ttest_mean([0, 1, 1])
    >>> tstat; pval; df
    array([ 9.72165021,  2.13019609,  0.52422632])
    array([  1.58414212e-12,   3.87842808e-02,   6.02752170e-01])
    44.0

    # if weights are integers, then asrepeats can be used

    >>> x1r = d1.asrepeats()
    >>> x1r.shape
    ...
    >>> stats.ttest_1samp(x1r, [0, 1, 1])
    ...

    """

    def __init__(self, data, weights=None, ddof=0):
        self.data = np.asarray(data)
        if weights is None:
            self.weights = np.ones(self.data.shape[0])
        else:
            self.weights = np.asarray(weights).astype(float)
            if len(self.weights.shape) > 1 and len(self.weights) > 1:
                self.weights = self.weights.squeeze()
        self.ddof = ddof

    @cache_readonly
    def sum_weights(self):
        """Sum of weights"""
        pass

    @cache_readonly
    def nobs(self):
        """alias for number of observations/cases, equal to sum of weights
        """
        pass

    @cache_readonly
    def sum(self):
        """weighted sum of data"""
        pass

    @cache_readonly
    def mean(self):
        """weighted mean of data"""
        pass

    @cache_readonly
    def demeaned(self):
        """data with weighted mean subtracted"""
        pass

    @cache_readonly
    def sumsquares(self):
        """weighted sum of squares of demeaned data"""
        pass

    def var_ddof(self, ddof=0):
        """variance of data given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        var : float, ndarray
            variance with denominator ``sum_weights - ddof``
        """
        pass

    def std_ddof(self, ddof=0):
        """standard deviation of data with given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        std : float, ndarray
            standard deviation with denominator ``sum_weights - ddof``
        """
        pass

    @cache_readonly
    def var(self):
        """variance with default degrees of freedom correction
        """
        pass

    @cache_readonly
    def _var(self):
        """variance without degrees of freedom correction

        used for statistical tests with controlled ddof
        """
        pass

    @cache_readonly
    def std(self):
        """standard deviation with default degrees of freedom correction
        """
        pass

    @cache_readonly
    def cov(self):
        """weighted covariance of data if data is 2 dimensional

        assumes variables in columns and observations in rows
        uses default ddof
        """
        pass

    @cache_readonly
    def corrcoef(self):
        """weighted correlation with default ddof

        assumes variables in columns and observations in rows
        """
        pass

    @cache_readonly
    def std_mean(self):
        """standard deviation of weighted mean
        """
        pass

    def quantile(self, probs, return_pandas=True):
        """
        Compute quantiles for a weighted sample.

        Parameters
        ----------
        probs : array_like
            A vector of probability points at which to calculate the
            quantiles.  Each element of `probs` should fall in [0, 1].
        return_pandas : bool
            If True, return value is a Pandas DataFrame or Series.
            Otherwise returns a ndarray.

        Returns
        -------
        quantiles : Series, DataFrame, or ndarray
            If `return_pandas` = True, returns one of the following:
              * data are 1d, `return_pandas` = True: a Series indexed by
                the probability points.
              * data are 2d, `return_pandas` = True: a DataFrame with
                the probability points as row index and the variables
                as column index.

            If `return_pandas` = False, returns an ndarray containing the
            same values as the Series/DataFrame.

        Notes
        -----
        To compute the quantiles, first, the weights are summed over
        exact ties yielding distinct data values y_1 < y_2 < ..., and
        corresponding weights w_1, w_2, ....  Let s_j denote the sum
        of the first j weights, and let W denote the sum of all the
        weights.  For a probability point p, if pW falls strictly
        between s_j and s_{j+1} then the estimated quantile is
        y_{j+1}.  If pW = s_j then the estimated quantile is (y_j +
        y_{j+1})/2.  If pW < p_1 then the estimated quantile is y_1.

        References
        ----------
        SAS documentation for weighted quantiles:

        https://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_univariate_sect028.htm
        """
        pass

    def tconfint_mean(self, alpha=0.05, alternative='two-sided'):
        """two-sided confidence interval for weighted mean of data

        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
        pass

    def zconfint_mean(self, alpha=0.05, alternative='two-sided'):
        """two-sided confidence interval for weighted mean of data

        Confidence interval is based on normal distribution.
        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
        pass

    def ttest_mean(self, value=0, alternative='two-sided'):
        """ttest of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following

        - 'two-sided': H1: mean not equal to value
        - 'larger' :   H1: mean larger than value
        - 'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following:

              - 'two-sided': H1: mean not equal to value (default)
              - 'larger' :   H1: mean larger than value
              - 'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float

        """
        pass

    def ttost_mean(self, low, upp):
        """test of (non-)equivalence of one sample

        TOST: two one-sided t tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1, df1 : tuple
            test statistic, pvalue and degrees of freedom for lower threshold
            test
        t2, pv2, df2 : tuple
            test statistic, pvalue and degrees of freedom for upper threshold
            test

        """
        pass

    def ztest_mean(self, value=0, alternative='two-sided'):
        """z-test of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following
        'two-sided': H1: mean not equal to value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test

        Notes
        -----
        This uses the same degrees of freedom correction as the t-test in the
        calculation of the standard error of the mean, i.e it uses
        `(sum_weights - 1)` instead of `sum_weights` in the denominator.
        See Examples below for the difference.

        Examples
        --------

        z-test on a proportion, with 20 observations, 15 of those are our event

        >>> import statsmodels.api as sm
        >>> x1 = [0, 1]
        >>> w1 = [5, 15]
        >>> d1 = sm.stats.DescrStatsW(x1, w1)
        >>> d1.ztest_mean(0.5)
        (2.5166114784235836, 0.011848940928347452)

        This differs from the proportions_ztest because of the degrees of
        freedom correction:
        >>> sm.stats.proportions_ztest(15, 20.0, value=0.5)
        (2.5819888974716112, 0.009823274507519247).

        We can replicate the results from ``proportions_ztest`` if we increase
        the weights to have artificially one more observation:

        >>> sm.stats.DescrStatsW(x1, np.array(w1)*21./20).ztest_mean(0.5)
        (2.5819888974716116, 0.0098232745075192366)
        """
        pass

    def ztost_mean(self, low, upp):
        """test of (non-)equivalence of one sample, based on z-test

        TOST: two one-sided z-tests

        null hypothesis:  m < low or m > upp
        alternative hypothesis:  low < m < upp

        where m is the expected value of the sample (mean of the population).

        If the pvalue is smaller than a threshold, say 0.05, then we reject the
        hypothesis that the expected value of the sample (mean of the
        population) is outside of the interval given by thresholds low and upp.

        Parameters
        ----------
        low, upp : float
            equivalence interval low < mean < upp

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple
            test statistic and p-value for lower threshold test
        t2, pv2 : tuple
            test statistic and p-value for upper threshold test

        """
        pass

    def get_compare(self, other, weights=None):
        """return an instance of CompareMeans with self and other

        Parameters
        ----------
        other : array_like or instance of DescrStatsW
            If array_like then this creates an instance of DescrStatsW with
            the given weights.
        weights : None or array
            weights are only used if other is not an instance of DescrStatsW

        Returns
        -------
        cm : instance of CompareMeans
            the instance has self attached as d1 and other as d2.

        See Also
        --------
        CompareMeans

        """
        pass

    def asrepeats(self):
        """get array that has repeats given by floor(weights)

        observations with weight=0 are dropped

        """
        pass

def _tstat_generic(value1, value2, std_diff, dof, alternative, diff=0):
    """generic ttest based on summary statistic

    The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff

    and is assumed to be t-distributed with ``dof`` degrees of freedom.

    Parameters
    ----------
    value1 : float or ndarray
        Value, for example mean, of the first sample.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std_diff : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    diff : float
        value of difference ``value1 - value2`` under the null hypothesis

    Returns
    -------
    tstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        t-distributed with ``df`` degrees of freedom.
    """
    pass

def _tconfint_generic(mean, std_mean, dof, alpha, alternative):
    """generic t-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    lower : float or ndarray
        Lower confidence limit. This is -inf for the one-sided alternative
        "smaller".
    upper : float or ndarray
        Upper confidence limit. This is inf for the one-sided alternative
        "larger".
    """
    pass

def _zstat_generic(value1, value2, std_diff, alternative, diff=0):
    """generic (normal) z-test based on summary statistic

    The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff

    and is assumed to be normally distributed.

    Parameters
    ----------
    value1 : float or ndarray
        Value, for example mean, of the first sample.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std_diff : float or ndarray
        Standard error of the difference value1 - value2
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    diff : float
        value of difference ``value1 - value2`` under the null hypothesis

    Returns
    -------
    tstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        t-distributed with ``df`` degrees of freedom.
    """
    pass

def _zstat_generic2(value, std, alternative):
    """generic (normal) z-test based on summary statistic

    The test statistic is :
        zstat = value / std

    and is assumed to be normally distributed with standard deviation ``std``.

    Parameters
    ----------
    value : float or ndarray
        Value of a sample statistic, for example mean.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std : float or ndarray
        Standard error of the sample statistic value.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    zstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        normally distributed.
    """
    pass

def _zconfint_generic(mean, std_mean, alpha, alternative):
    """generic normal-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    lower : float or ndarray
        Lower confidence limit. This is -inf for the one-sided alternative
        "smaller".
    upper : float or ndarray
        Upper confidence limit. This is inf for the one-sided alternative
        "larger".
    """
    pass

class CompareMeans:
    """class for two sample comparison

    The tests and the confidence interval work for multi-endpoint comparison:
    If d1 and d2 have the same number of rows, then each column of the data
    in d1 is compared with the corresponding column in d2.

    Parameters
    ----------
    d1, d2 : instances of DescrStatsW

    Notes
    -----
    The result for the statistical tests and the confidence interval are
    independent of the user specified ddof.

    TODO: Extend to any number of groups or write a version that works in that
    case, like in SAS and SPSS.

    """

    def __init__(self, d1, d2):
        """assume d1, d2 hold the relevant attributes

        """
        self.d1 = d1
        self.d2 = d2

    @classmethod
    def from_data(cls, data1, data2, weights1=None, weights2=None, ddof1=0, ddof2=0):
        """construct a CompareMeans object from data

        Parameters
        ----------
        data1, data2 : array_like, 1-D or 2-D
            compared datasets
        weights1, weights2 : None or 1-D ndarray
            weights for each observation of data1 and data2 respectively,
            with same length as zero axis of corresponding dataset.
        ddof1, ddof2 : int
            default ddof1=0, ddof2=0, degrees of freedom for data1,
            data2 respectively.

        Returns
        -------
        A CompareMeans instance.

        """
        pass

    def summary(self, use_t=True, alpha=0.05, usevar='pooled', value=0):
        """summarize the results of the hypothesis test

        Parameters
        ----------
        use_t : bool, optional
            if use_t is True, then t test results are returned
            if use_t is False, then z test results are returned
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is
            assumed to be the same. If ``unequal``, then the variance of
            Welch ttest will be used, and the degrees of freedom are those
            of Satterthwaite if ``use_t`` is True.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        smry : SimpleTable

        """
        pass

    @cache_readonly
    def std_meandiff_pooledvar(self):
        """variance assuming equal variance in both data sets

        """
        pass

    def dof_satt(self):
        """degrees of freedom of Satterthwaite for unequal variance
        """
        pass

    def ttest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        """ttest for the null hypothesis of identical means

        this should also be the same as onewaygls, except for ddof differences

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used
        value : float
            difference between the means under the Null hypothesis.


        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test
        df : int or float
            degrees of freedom used in the t-test

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        pass

    def ztest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        """z-test for the null hypothesis of identical means

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then the standard deviations of the samples may
            be different.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the z-test

        """
        pass

    def tconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        pass

    def zconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
        pass

    def ttost_ind(self, low, upp, usevar='pooled'):
        """
        test of equivalence for two independent samples, base on t-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        pass

    def ztost_ind(self, low, upp, usevar='pooled'):
        """
        test of equivalence for two independent samples, based on z-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
        pass

def ttest_ind(x1, x2, alternative='two-sided', usevar='pooled', weights=(None, None), value=0):
    """ttest independent sample

    Convenience function that uses the classes and throws away the intermediate
    results,
    compared to scipy stats: drops axis option, adds alternative, usevar, and
    weights option.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' (default): H1: difference in means not equal to value
           * 'larger' :   H1: difference in means larger than value
           * 'smaller' :  H1: difference in means smaller than value

    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then Welch ttest with Satterthwait degrees
        of freedom is used
    weights : tuple of None or ndarrays
        Case weights for the two samples. For details on weights see
        ``DescrStatsW``
    value : float
        difference between the means under the Null hypothesis.


    Returns
    -------
    tstat : float
        test statistic
    pvalue : float
        pvalue of the t-test
    df : int or float
        degrees of freedom used in the t-test

    """
    pass

def ttost_ind(x1, x2, low, upp, usevar='pooled', weights=(None, None), transform=None):
    """test of (non-)equivalence for two independent samples

    TOST: two one-sided t tests

    null hypothesis:  m1 - m2 < low or m1 - m2 > upp
    alternative hypothesis:  low < m1 - m2 < upp

    where m1, m2 are the means, expected values of the two samples.

    If the pvalue is smaller than a threshold, say 0.05, then we reject the
    hypothesis that the difference between the two samples is larger than the
    the thresholds given by low and upp.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    low, upp : float
        equivalence interval low < m1 - m2 < upp
    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then Welch ttest with Satterthwait degrees
        of freedom is used
    weights : tuple of None or ndarrays
        Case weights for the two samples. For details on weights see
        ``DescrStatsW``
    transform : None or function
        If None (default), then the data is not transformed. Given a function,
        sample data and thresholds are transformed. If transform is log, then
        the equivalence interval is in ratio: low < m1 / m2 < upp

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
    The test rejects if the 2*alpha confidence interval for the difference
    is contained in the ``(low, upp)`` interval.

    This test works also for multi-endpoint comparisons: If d1 and d2
    have the same number of columns, then each column of the data in d1 is
    compared with the corresponding column in d2. This is the same as
    comparing each of the corresponding columns separately. Currently no
    multi-comparison correction is used. The raw p-values reported here can
    be correction with the functions in ``multitest``.

    """
    pass

def ttost_paired(x1, x2, low, upp, transform=None, weights=None):
    """test of (non-)equivalence for two dependent, paired sample

    TOST: two one-sided t tests

    null hypothesis:  md < low or md > upp
    alternative hypothesis:  low < md < upp

    where md is the mean, expected value of the difference x1 - x2

    If the pvalue is smaller than a threshold,say 0.05, then we reject the
    hypothesis that the difference between the two samples is larger than the
    the thresholds given by low and upp.

    Parameters
    ----------
    x1 : array_like
        first of the two independent samples
    x2 : array_like
        second of the two independent samples
    low, upp : float
        equivalence interval low < mean of difference < upp
    weights : None or ndarray
        case weights for the two samples. For details on weights see
        ``DescrStatsW``
    transform : None or function
        If None (default), then the data is not transformed. Given a function
        sample data and thresholds are transformed. If transform is log the
        the equivalence interval is in ratio: low < x1 / x2 < upp

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1, df1 : tuple
        test statistic, pvalue and degrees of freedom for lower threshold test
    t2, pv2, df2 : tuple
        test statistic, pvalue and degrees of freedom for upper threshold test

    """
    pass

def ztest(x1, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0):
    """test for mean based on normal distribution, one or two samples

    In the case of two samples, the samples are assumed to be independent.

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples
    x2 : array_like, 1-D or 2-D
        second of the two independent samples
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: difference in means not equal to value (default)
           'larger' :   H1: difference in means larger than value
           'smaller' :  H1: difference in means smaller than value

    usevar : str, 'pooled' or 'unequal'
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. If ``unequal``, then the standard deviation of the sample is
        assumed to be different.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Returns
    -------
    tstat : float
        test statistic
    pvalue : float
        pvalue of the t-test

    Notes
    -----
    usevar can be pooled or unequal in two sample case

    """
    pass

def zconfint(x1, x2=None, value=0, alpha=0.05, alternative='two-sided', usevar='pooled', ddof=1.0):
    """confidence interval based on normal distribution z-test

    Parameters
    ----------
    x1 : array_like, 1-D or 2-D
        first of the two independent samples, see notes for 2-D case
    x2 : array_like, 1-D or 2-D
        second of the two independent samples, see notes for 2-D case
    value : float
        In the one sample case, value is the mean of x1 under the Null
        hypothesis.
        In the two sample case, value is the difference between mean of x1 and
        mean of x2 under the Null hypothesis. The test statistic is
        `x1_mean - x2_mean - value`.
    usevar : str, 'pooled'
        Currently, only 'pooled' is implemented.
        If ``pooled``, then the standard deviation of the samples is assumed to be
        the same. see CompareMeans.ztest_ind for different options.
    ddof : int
        Degrees of freedom use in the calculation of the variance of the mean
        estimate. In the case of comparing means this is one, however it can
        be adjusted for testing other statistics (proportion, correlation)

    Notes
    -----
    checked only for 1 sample case

    usevar not implemented, is always pooled in two sample case

    ``value`` shifts the confidence interval so it is centered at
    `x1_mean - x2_mean - value`

    See Also
    --------
    ztest
    CompareMeans

    """
    pass

def ztost(x1, low, upp, x2=None, usevar='pooled', ddof=1.0):
    """Equivalence test based on normal distribution

    Parameters
    ----------
    x1 : array_like
        one sample or first sample for 2 independent samples
    low, upp : float
        equivalence interval low < m1 - m2 < upp
    x1 : array_like or None
        second sample for 2 independent samples test. If None, then a
        one-sample test is performed.
    usevar : str, 'pooled'
        If `pooled`, then the standard deviation of the samples is assumed to be
        the same. Only `pooled` is currently implemented.

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