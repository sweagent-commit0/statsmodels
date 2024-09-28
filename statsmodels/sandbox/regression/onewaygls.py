"""
F test for null hypothesis that coefficients in several regressions are the same

* implemented by creating groupdummies*exog and testing appropriate contrast
  matrices
* similar to test for structural change in all variables at predefined break points
* allows only one group variable
* currently tests for change in all exog variables
* allows for heterogscedasticity, error variance varies across groups
* does not work if there is a group with only a single observation

TODO
----

* generalize anova structure,
  - structural break in only some variables
  - compare structural breaks in several exog versus constant only
  - fast way to construct comparisons
* print anova style results
* add all pairwise comparison tests (DONE) with and without Bonferroni correction
* add additional test, likelihood-ratio, lagrange-multiplier, wald ?
* test for heteroscedasticity, equality of variances
  - how?
  - like lagrange-multiplier in stattools heteroscedasticity tests
* permutation or bootstrap test statistic or pvalues


References
----------

Greene: section 7.4 Modeling and Testing for a Structural Break
   is not the same because I use a different normalization, which looks easier
   for more than 2 groups/subperiods

after looking at Greene:
* my version assumes that all groups are large enough to estimate the coefficients
* in sections 7.4.2 and 7.5.3, predictive tests can also be used when there are
  insufficient (nobs<nvars) observations in one group/subperiods
  question: can this be used to test structural change for last period?
        cusum test but only for current period,
        in general cusum is better done with recursive ols
        check other references again for this, there was one for non-recursive
        calculation of cusum (if I remember correctly)
* Greene 7.4.4: with unequal variances Greene mentions Wald test, but where
  size of test might not be very good
  no mention of F-test based on GLS, is there a reference for what I did?
  alternative: use Wald test with bootstrap pvalues?


Created on Sat Mar 27 01:48:01 2010
Author: josef-pktd
"""
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS

class OneWayLS:
    """Class to test equality of regression coefficients across groups

    This class performs tests whether the linear regression coefficients are
    the same across pre-specified groups. This can be used to test for
    structural breaks at given change points, or for ANOVA style analysis of
    differences in the effect of explanatory variables across groups.

    Notes
    -----
    The test is implemented by regression on the original pooled exogenous
    variables and on group dummies times the exogenous regressors.

    y_i = X_i beta_i + u_i  for all groups i

    The test is for the null hypothesis: beta_i = beta for all i
    against the alternative that at least one beta_i is different.

    By default it is assumed that all u_i have the same variance. If the
    keyword option het is True, then it is assumed that the variance is
    group specific. This uses WLS with weights given by the standard errors
    from separate regressions for each group.
    Note: het=True is not sufficiently tested

    The F-test assumes that the errors are normally distributed.



    original question from mailing list for equality of coefficients
    across regressions, and example in Stata FAQ

    *testing*:

    * if constant is the only regressor then the result for the F-test is
      the same as scipy.stats.f_oneway
      (which in turn is verified against NIST for not badly scaled problems)
    * f-test for simple structural break is the same as in original script
    * power and size of test look ok in examples
    * not checked/verified for heteroskedastic case
      - for constant only: ftest result is the same with WLS as with OLS - check?

    check: I might be mixing up group names (unique)
           and group id (integers in arange(ngroups)
           not tested for groups that are not arange(ngroups)
           make sure groupnames are always consistently sorted/ordered
           Fixed for getting the results, but groups are not printed yet, still
           inconsistent use for summaries of results.
    """

    def __init__(self, y, x, groups=None, het=False, data=None, meta=None):
        if groups is None:
            raise ValueError('use OLS if there are no groups')
        if data:
            y = data[y]
            x = [data[v] for v in x]
            try:
                groups = data[groups]
            except [KeyError, ValueError]:
                pass
        self.endog = np.asarray(y)
        self.exog = np.asarray(x)
        if self.exog.ndim == 1:
            self.exog = self.exog[:, None]
        self.groups = np.asarray(groups)
        self.het = het
        self.groupsint = None
        if np.issubdtype(self.groups.dtype, int):
            self.unique = np.unique(self.groups)
            if (self.unique == np.arange(len(self.unique))).all():
                self.groupsint = self.groups
        if self.groupsint is None:
            self.unique, self.groupsint = np.unique(self.groups, return_inverse=True)
        self.uniqueint = np.arange(len(self.unique))

    def fitbygroups(self):
        """Fit OLS regression for each group separately.

        Returns
        -------
        results are attached

        olsbygroup : dictionary of result instance
            the returned regression results for each group
        sigmabygroup : array (ngroups,) (this should be called sigma2group ??? check)
            mse_resid for each group
        weights : array (nobs,)
            standard deviation of group extended to the original observations. This can
            be used as weights in WLS for group-wise heteroscedasticity.



        """
        pass

    def fitjoint(self):
        """fit a joint fixed effects model to all observations

        The regression results are attached as `lsjoint`.

        The contrasts for overall and pairwise tests for equality of coefficients are
        attached as a dictionary `contrasts`. This also includes the contrasts for the test
        that the coefficients of a level are zero. ::

        >>> res.contrasts.keys()
        [(0, 1), 1, 'all', 3, (1, 2), 2, (1, 3), (2, 3), (0, 3), (0, 2)]

        The keys are based on the original names or labels of the groups.

        TODO: keys can be numpy scalars and then the keys cannot be sorted



        """
        pass

    def fitpooled(self):
        """fit the pooled model, which assumes there are no differences across groups
        """
        pass

    def ftest_summary(self):
        """run all ftests on the joint model

        Returns
        -------
        fres : str
           a string that lists the results of all individual f-tests
        summarytable : list of tuples
           contains (pair, (fvalue, pvalue,df_denom, df_num)) for each f-test

        Note
        ----
        This are the raw results and not formatted for nice printing.

        """
        pass

    def print_summary(self, res):
        """printable string of summary

        """
        pass

    def lr_test(self):
        """
        generic likelihood ratio test between nested models

            \\begin{align}
            D & = -2(\\ln(\\text{likelihood for null model}) - \\ln(\\text{likelihood for alternative model})) \\\\
            & = -2\\ln\\left( \\frac{\\text{likelihood for null model}}{\\text{likelihood for alternative model}} \\right).
            \\end{align}

        is distributed as chisquare with df equal to difference in number of parameters or equivalently
        difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        """
        pass