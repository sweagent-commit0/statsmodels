"""Canonical correlation analysis

author: Yichuan Liu
"""
import numpy as np
from numpy.linalg import svd
import scipy
import pandas as pd
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from .multivariate_ols import multivariate_stats

class CanCorr(Model):
    """
    Canonical correlation analysis using singular value decomposition

    For matrices exog=x and endog=y, find projections x_cancoef and y_cancoef
    such that:

        x1 = x * x_cancoef, x1' * x1 is identity matrix
        y1 = y * y_cancoef, y1' * y1 is identity matrix

    and the correlation between x1 and y1 is maximized.

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    cancorr : ndarray
        The canonical correlation values
    y_cancoef : ndarray
        The canonical coefficients for endog
    x_cancoef : ndarray
        The canonical coefficients for exog

    References
    ----------
    .. [*] http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
    .. [*] http://www.csun.edu/~ata20315/psy524/docs/Psy524%20Lecture%208%20CC.pdf
    .. [*] http://www.mathematica-journal.com/2014/06/canonical-correlation-analysis/
    """

    def __init__(self, endog, exog, tolerance=1e-08, missing='none', hasconst=None, **kwargs):
        super(CanCorr, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)
        self._fit(tolerance)

    def _fit(self, tolerance=1e-08):
        """Fit the model

        A ValueError is raised if there are singular values smaller than the
        tolerance. The treatment of singular arrays might change in future.

        Parameters
        ----------
        tolerance : float
            eigenvalue tolerance, values smaller than which is considered 0
        """
        pass

    def corr_test(self):
        """Approximate F test
        Perform multivariate statistical tests of the hypothesis that
        there is no canonical correlation between endog and exog.
        For each canonical correlation, testing its significance based on
        Wilks' lambda.

        Returns
        -------
        CanCorrTestResults instance
        """
        pass

class CanCorrTestResults:
    """
    Canonical correlation results class

    Attributes
    ----------
    stats : DataFrame
        Contain statistical tests results for each canonical correlation
    stats_mv : DataFrame
        Contain the multivariate statistical tests results
    """

    def __init__(self, stats, stats_mv):
        self.stats = stats
        self.stats_mv = stats_mv

    def __str__(self):
        return self.summary().__str__()