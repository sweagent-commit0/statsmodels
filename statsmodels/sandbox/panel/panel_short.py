"""Panel data analysis for short T and large N

Created on Sat Dec 17 19:32:00 2011

Author: Josef Perktold
License: BSD-3


starting from scratch before looking at references again
just a stub to get the basic structure for group handling
target outsource as much as possible for reuse

Notes
-----

this is the basic version using a loop over individuals which will be more
widely applicable. Depending on the special cases, there will be faster
implementations possible (sparse, kroneker, ...)

the only two group specific methods or get_within_cov and whiten

"""
import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted

def sum_outer_product_loop(x, group_iter):
    """sum outerproduct dot(x_i, x_i.T) over individuals

    loop version

    """
    pass

def sum_outer_product_balanced(x, n_groups):
    """sum outerproduct dot(x_i, x_i.T) over individuals

    where x_i is (nobs_i, 1), and result is (nobs_i, nobs_i)

    reshape-dot version, for x.ndim=1 only

    """
    pass

def whiten_individuals_loop(x, transform, group_iter):
    """apply linear transform for each individual

    loop version
    """
    pass

class ShortPanelGLS2:
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group):
        self.endog = endog
        self.exog = exog
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups

class ShortPanelGLS(GLS):
    """Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    """

    def __init__(self, endog, exog, group, sigma_i=None):
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups
        nobs_i = len(endog) / self.n_groups
        if sigma_i is None:
            sigma_i = np.eye(int(nobs_i))
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        super(self.__class__, self).__init__(endog, exog, sigma=None)

    def fit_iterative(self, maxiter=3):
        """
        Perform an iterative two-step procedure to estimate the GLS model.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) once does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
        pass