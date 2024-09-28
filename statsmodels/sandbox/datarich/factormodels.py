"""
Created on Sun Nov 14 08:21:41 2010

Author: josef-pktd
License: BSD (3-clause)
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut

class FactorModelUnivariate:
    """

    Todo:
    check treatment of const, make it optional ?
        add hasconst (0 or 1), needed when selecting nfact+hasconst
    options are arguments in calc_factors, should be more public instead
    cross-validation is slow for large number of observations
    """

    def __init__(self, endog, exog):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)

    def calc_factors(self, x=None, keepdim=0, addconst=True):
        """get factor decomposition of exogenous variables

        This uses principal component analysis to obtain the factors. The number
        of factors kept is the maximum that will be considered in the regression.
        """
        pass

    def fit_find_nfact(self, maxfact=None, skip_crossval=True, cv_iter=None):
        """estimate the model and selection criteria for up to maxfact factors

        The selection criteria that are calculated are AIC, BIC, and R2_adj. and
        additionally cross-validation prediction error sum of squares if `skip_crossval`
        is false. Cross-validation is not used by default because it can be
        time consuming to calculate.

        By default the cross-validation method is Leave-one-out on the full dataset.
        A different cross-validation sample can be specified as an argument to
        cv_iter.

        Results are attached in `results_find_nfact`



        """
        pass

    def summary_find_nfact(self):
        """provides a summary for the selection of the number of factors

        Returns
        -------
        sumstr : str
            summary of the results for selecting the number of factors

        """
        pass
if __name__ == '__main__':
    examples = [1]
    if 1 in examples:
        nobs = 500
        f0 = np.c_[np.random.normal(size=(nobs, 2)), np.ones((nobs, 1))]
        f2xcoef = np.c_[np.repeat(np.eye(2), 2, 0), np.arange(4)[::-1]].T
        f2xcoef = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
        f2xcoef = np.array([[0.1, 3.0, 1.0, 0.0], [0.0, 0.0, 1.5, 0.1], [3.0, 2.0, 1.0, 0.0]])
        x0 = np.dot(f0, f2xcoef)
        x0 += 0.1 * np.random.normal(size=x0.shape)
        ytrue = np.dot(f0, [1.0, 1.0, 1.0])
        y0 = ytrue + 0.1 * np.random.normal(size=ytrue.shape)
        mod = FactorModelUnivariate(y0, x0)
        print(mod.summary_find_nfact())
        print('with cross validation - slower')
        mod.fit_find_nfact(maxfact=None, skip_crossval=False, cv_iter=None)
        print(mod.summary_find_nfact())