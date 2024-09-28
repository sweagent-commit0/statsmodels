"""
Created on Mon Oct  5 12:36:54 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import special
from statsmodels.stats.base import Holder

def _noncentrality_chisquare(chi2_stat, df, alpha=0.05):
    """noncentrality parameter for chi-square statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    chi2_stat : float
        Chisquare-statistic, for example from a hypothesis test
    df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of
        Noncentrality Parameters.”
        Canadian Journal of Statistics 21 (1): 45–57.
        https://doi.org/10.2307/3315657.

    .. [2] Li, Qizhai, Junjian Zhang, and Shuai Dai. 2009. “On Estimating the
        Non-Centrality Parameter of a Chi-Squared Distribution.”
        Statistics & Probability Letters 79 (1): 98–104.
        https://doi.org/10.1016/j.spl.2008.07.025.

    """
    pass

def _noncentrality_f(f_stat, df1, df2, alpha=0.05):
    """noncentrality parameter for f statistic

    `nc` is zero-truncated umvue

    Parameters
    ----------
    fstat : float
        f-statistic, for example from a hypothesis test
        df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of
       Noncentrality Parameters.” Canadian Journal of Statistics 21 (1): 45–57.
       https://doi.org/10.2307/3315657.
    """
    pass

def _noncentrality_t(t_stat, df, alpha=0.05):
    """noncentrality parameter for t statistic

    Parameters
    ----------
    fstat : float
        f-statistic, for example from a hypothesis test
        df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Hedges, Larry V. 2016. “Distribution Theory for Glass’s Estimator of
       Effect Size and Related Estimators:”
       Journal of Educational Statistics, November.
       https://doi.org/10.3102/10769986006002107.

    """
    pass