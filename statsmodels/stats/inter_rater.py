"""Inter Rater Agreement

contains
--------
fleiss_kappa
cohens_kappa

aggregate_raters:
    helper function to get data into fleiss_kappa format
to_table:
    helper function to create contingency table, can be used for cohens_kappa

Created on Thu Dec 06 22:57:56 2012
Author: Josef Perktold
License: BSD-3

References
----------
Wikipedia: kappa's initially based on these two pages
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    https://en.wikipedia.org/wiki/Cohen's_kappa
SAS-Manual : formulas for cohens_kappa, especially variances
see also R package irr

TODO
----
standard errors and hypothesis tests for fleiss_kappa
other statistics and tests,
   in R package irr, SAS has more
inconsistent internal naming, changed variable names as I added more
   functionality
convenience functions to create required data format from raw data
   DONE

"""
import numpy as np
from scipy import stats

class ResultsBunch(dict):
    template = '%r'

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self
        self._initialize()

    def __str__(self):
        return self.template % self

def _int_ifclose(x, dec=1, width=4):
    """helper function for creating result string for int or float

    only dec=1 and width=4 is implemented

    Parameters
    ----------
    x : int or float
        value to format
    dec : 1
        number of decimals to print if x is not an integer
    width : 4
        width of string

    Returns
    -------
    xint : int or float
        x is converted to int if it is within 1e-14 of an integer
    x_string : str
        x formatted as string, either '%4d' or '%4.1f'

    """
    pass

def aggregate_raters(data, n_cat=None):
    """convert raw data with shape (subject, rater) to (subject, cat_counts)

    brings data into correct format for fleiss_kappa

    bincount will raise exception if data cannot be converted to integer.

    Parameters
    ----------
    data : array_like, 2-Dim
        data containing category assignment with subjects in rows and raters
        in columns.
    n_cat : None or int
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are already
        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the
        returned array may contain columns with zero count, if no subject
        has been categorized with this level.

    Returns
    -------
    arr : nd_array, (n_rows, n_cat)
        Contains counts of raters that assigned a category level to individuals.
        Subjects are in rows, category levels in columns.
    categories : nd_array, (n_category_levels,)
        Contains the category levels.

    """
    pass

def to_table(data, bins=None):
    """convert raw data with shape (subject, rater) to (rater1, rater2)

    brings data into correct format for cohens_kappa

    Parameters
    ----------
    data : array_like, 2-Dim
        data containing category assignment with subjects in rows and raters
        in columns.
    bins : None, int or tuple of array_like
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are already
        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the
        returned array may contain columns with zero count, if no subject
        has been categorized with this level.
        If bins are a tuple of two array_like, then the bins are directly used
        by ``numpy.histogramdd``. This is useful if we want to merge categories.

    Returns
    -------
    arr : nd_array, (n_cat, n_cat)
        Contingency table that contains counts of category level with rater1
        in rows and rater2 in columns.

    Notes
    -----
    no NaN handling, delete rows with missing values

    This works also for more than two raters. In that case the dimension of
    the resulting contingency table is the same as the number of raters
    instead of 2-dimensional.

    """
    pass

def fleiss_kappa(table, method='fleiss'):
    """Fleiss' and Randolph's kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns. Convert raw data
        into this format by using
        :func:`statsmodels.stats.inter_rater.aggregate_raters`
    method : str
        Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.

    Returns
    -------
    kappa : float
        Fleiss's or Randolph's kappa statistic for inter rater agreement

    Notes
    -----
    no variance or hypothesis tests yet

    Interrater agreement measures like Fleiss's kappa measure agreement relative
    to chance agreement. Different authors have proposed ways of defining
    these chance agreements. Fleiss' is based on the marginal sample distribution
    of categories, while Randolph uses a uniform distribution of categories as
    benchmark. Warrens (2010) showed that Randolph's kappa is always larger or
    equal to Fleiss' kappa. Under some commonly observed condition, Fleiss' and
    Randolph's kappa provide lower and upper bounds for two similar kappa_like
    measures by Light (1971) and Hubert (1977).

    References
    ----------
    Wikipedia https://en.wikipedia.org/wiki/Fleiss%27_kappa

    Fleiss, Joseph L. 1971. "Measuring Nominal Scale Agreement among Many
    Raters." Psychological Bulletin 76 (5): 378-82.
    https://doi.org/10.1037/h0031619.

    Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater
    K [free]): An Alternative to Fleiss' Fixed-Marginal Multirater Kappa."
    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005
    https://eric.ed.gov/?id=ED490661

    Warrens, Matthijs J. 2010. "Inequalities between Multi-Rater Kappas."
    Advances in Data Analysis and Classification 4 (4): 271-86.
    https://doi.org/10.1007/s11634-010-0073-4.
    """
    pass

def cohens_kappa(table, weights=None, return_results=True, wt=None):
    """Compute Cohen's kappa with variance and equal-zero test

    Parameters
    ----------
    table : array_like, 2-Dim
        square array with results of two raters, one rater in rows, second
        rater in columns
    weights : array_like
        The interpretation of weights depends on the wt argument.
        If both are None, then the simple kappa is computed.
        see wt for the case when wt is not None
        If weights is two dimensional, then it is directly used as a weight
        matrix. For computing the variance of kappa, the maximum of the
        weights is assumed to be smaller or equal to one.
        TODO: fix conflicting definitions in the 2-Dim case for
    wt : {None, str}
        If wt and weights are None, then the simple kappa is computed.
        If wt is given, but weights is None, then the weights are set to
        be [0, 1, 2, ..., k].
        If weights is a one-dimensional array, then it is used to construct
        the weight matrix given the following options.

        wt in ['linear', 'ca' or None] : use linear weights, Cicchetti-Allison
            actual weights are linear in the score "weights" difference
        wt in ['quadratic', 'fc'] : use linear weights, Fleiss-Cohen
            actual weights are squared in the score "weights" difference
        wt = 'toeplitz' : weight matrix is constructed as a toeplitz matrix
            from the one dimensional weights.

    return_results : bool
        If True (default), then an instance of KappaResults is returned.
        If False, then only kappa is computed and returned.

    Returns
    -------
    results or kappa
        If return_results is True (default), then a results instance with all
        statistics is returned
        If return_results is False, then only kappa is calculated and returned.

    Notes
    -----
    There are two conflicting definitions of the weight matrix, Wikipedia
    versus SAS manual. However, the computation are invariant to rescaling
    of the weights matrix, so there is no difference in the results.

    Weights for 'linear' and 'quadratic' are interpreted as scores for the
    categories, the weights in the computation are based on the pairwise
    difference between the scores.
    Weights for 'toeplitz' are a interpreted as weighted distance. The distance
    only depends on how many levels apart two entries in the table are but
    not on the levels themselves.

    example:

    weights = '0, 1, 2, 3' and wt is either linear or toeplitz means that the
    weighting only depends on the simple distance of levels.

    weights = '0, 0, 1, 1' and wt = 'linear' means that the first two levels
    are zero distance apart and the same for the last two levels. This is
    the sample as forming two aggregated levels by merging the first two and
    the last two levels, respectively.

    weights = [0, 1, 2, 3] and wt = 'quadratic' is the same as squaring these
    weights and using wt = 'toeplitz'.

    References
    ----------
    Wikipedia
    SAS Manual

    """
    pass
_kappa_template = '                  %(kind)s Kappa Coefficient\n              --------------------------------\n              Kappa                     %(kappa)6.4f\n              ASE                       %(std_kappa)6.4f\n            %(alpha_ci)s%% Lower Conf Limit      %(kappa_low)6.4f\n            %(alpha_ci)s%% Upper Conf Limit      %(kappa_upp)6.4f\n\n                 Test of H0: %(kind)s Kappa = 0\n\n              ASE under H0              %(std_kappa0)6.4f\n              Z                         %(z_value)6.4f\n              One-sided Pr >  Z         %(pvalue_one_sided)6.4f\n              Two-sided Pr > |Z|        %(pvalue_two_sided)6.4f\n'
'\n                   Weighted Kappa Coefficient\n              --------------------------------\n              Weighted Kappa            0.4701\n              ASE                       0.1457\n              95% Lower Conf Limit      0.1845\n              95% Upper Conf Limit      0.7558\n\n               Test of H0: Weighted Kappa = 0\n\n              ASE under H0              0.1426\n              Z                         3.2971\n              One-sided Pr >  Z         0.0005\n              Two-sided Pr > |Z|        0.0010\n'

class KappaResults(ResultsBunch):
    """Results for Cohen's kappa

    Attributes
    ----------
    kappa : cohen's kappa
    var_kappa : variance of kappa
    std_kappa : standard deviation of kappa
    alpha : one-sided probability for confidence interval
    kappa_low : lower (1-alpha) confidence limit
    kappa_upp : upper (1-alpha) confidence limit
    var_kappa0 : variance of kappa under H0: kappa=0
    std_kappa0 : standard deviation of kappa under H0: kappa=0
    z_value : test statistic for H0: kappa=0, is standard normal distributed
    pvalue_one_sided : one sided p-value for H0: kappa=0 and H1: kappa>0
    pvalue_two_sided : two sided p-value for H0: kappa=0 and H1: kappa!=0
    distribution_kappa : asymptotic normal distribution of kappa
    distribution_zero_null : asymptotic normal distribution of kappa under
        H0: kappa=0

    The confidence interval for kappa and the statistics for the test of
    H0: kappa=0 are based on the asymptotic normal distribution of kappa.

    """
    template = _kappa_template

    def __str__(self):
        return self.template % self