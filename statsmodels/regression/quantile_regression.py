"""
Quantile regression model

Model parameters are estimated using iterated reweighted least squares. The
asymptotic covariance matrix estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
"""
import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import RegressionModel, RegressionResults, RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning

class QuantReg(RegressionModel):
    """Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.

    Parameters
    ----------
    endog : array or dataframe
        endogenous/response variable
    exog : array or dataframe
        exogenous/explanatory variable(s)

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P.(1999). Applied Econometrics Using MATLAB,

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    """

    def __init__(self, endog, exog, **kwargs):
        self._check_kwargs(kwargs)
        super(QuantReg, self).__init__(endog, exog, **kwargs)

    def whiten(self, data):
        """
        QuantReg model whitener does nothing: returns data.
        """
        pass

    def fit(self, q=0.5, vcov='robust', kernel='epa', bandwidth='hsheather', max_iter=1000, p_tol=1e-06, **kwargs):
        """
        Solve by Iterative Weighted Least Squares

        Parameters
        ----------
        q : float
            Quantile must be strictly between 0 and 1
        vcov : str, method used to calculate the variance-covariance matrix
            of the parameters. Default is ``robust``:

            - robust : heteroskedasticity robust standard errors (as suggested
              in Greene 6th edition)
            - iid : iid errors (as in Stata 12)

        kernel : str, kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzene

        bandwidth : str, Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        """
        pass
kernels = {}
kernels['biw'] = lambda u: 15.0 / 16 * (1 - u ** 2) ** 2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= 0.5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3.0 / 4 * (1 - u ** 2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['gau'] = norm.pdf
kernels['par'] = _parzen

class QuantRegResults(RegressionResults):
    """Results instance for the QuantReg model"""

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        pass