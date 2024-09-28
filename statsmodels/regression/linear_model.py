"""
This module implements standard regression models:

Generalized Least Squares (GLS)
Ordinary Least Squares (OLS)
Weighted Least Squares (WLS)
Generalized Least Squares with autoregressive error terms GLSAR(p)

Models are specified with an endogenous response variable and an
exogenous design matrix and are fit using their `fit` method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

D. C. Montgomery and E.A. Peck. "Introduction to Linear Regression
    Analysis." 2nd. Ed., Wiley, 1992.

Econometrics references for regression models:

R. Davidson and J.G. MacKinnon.  "Econometric Theory and Methods," Oxford,
    2004.

W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
"""
from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal, Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import InvalidTestWarning, ValueWarning
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
__docformat__ = 'restructuredtext en'
__all__ = ['GLS', 'WLS', 'OLS', 'GLSAR', 'PredictionResults', 'RegressionResultsWrapper']
_fit_regularized_doc = "\n        Return a regularized fit to a linear regression model.\n\n        Parameters\n        ----------\n        method : str\n            Either 'elastic_net' or 'sqrt_lasso'.\n        alpha : scalar or array_like\n            The penalty weight.  If a scalar, the same penalty weight\n            applies to all variables in the model.  If a vector, it\n            must have the same length as `params`, and contains a\n            penalty weight for each coefficient.\n        L1_wt : scalar\n            The fraction of the penalty given to the L1 penalty term.\n            Must be between 0 and 1 (inclusive).  If 0, the fit is a\n            ridge fit, if 1 it is a lasso fit.\n        start_params : array_like\n            Starting values for ``params``.\n        profile_scale : bool\n            If True the penalized fit is computed using the profile\n            (concentrated) log-likelihood for the Gaussian model.\n            Otherwise the fit uses the residual sum of squares.\n        refit : bool\n            If True, the model is refit using only the variables that\n            have non-zero coefficients in the regularized fit.  The\n            refitted model is not regularized.\n        **kwargs\n            Additional keyword arguments that contain information used when\n            constructing a model using the formula interface.\n\n        Returns\n        -------\n        statsmodels.base.elastic_net.RegularizedResults\n            The regularized results.\n\n        Notes\n        -----\n        The elastic net uses a combination of L1 and L2 penalties.\n        The implementation closely follows the glmnet package in R.\n\n        The function that is minimized is:\n\n        .. math::\n\n            0.5*RSS/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)\n\n        where RSS is the usual regression sum of squares, n is the\n        sample size, and :math:`|*|_1` and :math:`|*|_2` are the L1 and L2\n        norms.\n\n        For WLS and GLS, the RSS is calculated using the whitened endog and\n        exog data.\n\n        Post-estimation results are based on the same data used to\n        select variables, hence may be subject to overfitting biases.\n\n        The elastic_net method uses the following keyword arguments:\n\n        maxiter : int\n            Maximum number of iterations\n        cnvrg_tol : float\n            Convergence threshold for line searches\n        zero_tol : float\n            Coefficients below this threshold are treated as zero.\n\n        The square root lasso approach is a variation of the Lasso\n        that is largely self-tuning (the optimal tuning parameter\n        does not depend on the standard deviation of the regression\n        errors).  If the errors are Gaussian, the tuning parameter\n        can be taken to be\n\n        alpha = 1.1 * np.sqrt(n) * norm.ppf(1 - 0.05 / (2 * p))\n\n        where n is the sample size and p is the number of predictors.\n\n        The square root lasso uses the following keyword arguments:\n\n        zero_tol : float\n            Coefficients below this threshold are treated as zero.\n\n        The cvxopt module is required to estimate model using the square root\n        lasso.\n\n        References\n        ----------\n        .. [*] Friedman, Hastie, Tibshirani (2008).  Regularization paths for\n           generalized linear models via coordinate descent.  Journal of\n           Statistical Software 33(1), 1-22 Feb 2010.\n\n        .. [*] A Belloni, V Chernozhukov, L Wang (2011).  Square-root Lasso:\n           pivotal recovery of sparse signals via conic programming.\n           Biometrika 98(4), 791-806. https://arxiv.org/pdf/1009.5689.pdf\n        "

def _get_sigma(sigma, nobs):
    """
    Returns sigma (matrix, nobs by nobs) for GLS and the inverse of its
    Cholesky decomposition.  Handles dimensions and checks integrity.
    If sigma is None, returns None, None. Otherwise returns sigma,
    cholsigmainv.
    """
    pass

class RegressionModel(base.LikelihoodModel):
    """
    Base class for linear regression models. Should not be directly called.

    Intended for subclassing.
    """

    def __init__(self, endog, exog, **kwargs):
        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self.pinv_wexog: Float64Array | None = None
        self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])

    def initialize(self):
        """Initialize model components."""
        pass

    @property
    def df_model(self):
        """
        The model degree of freedom.

        The dof is defined as the rank of the regressor matrix minus 1 if a
        constant is included.
        """
        pass

    @property
    def df_resid(self):
        """
        The residual degree of freedom.

        The dof is defined as the number of observations minus the rank of
        the regressor matrix.
        """
        pass

    def whiten(self, x):
        """
        Whiten method that must be overwritten by individual models.

        Parameters
        ----------
        x : array_like
            Data to be whitened.
        """
        pass

    def fit(self, method: Literal['pinv', 'qr']='pinv', cov_type: Literal['nonrobust', 'fixed scale', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'hac-panel', 'hac-groupsum', 'cluster']='nonrobust', cov_kwds=None, use_t: bool | None=None, **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str, optional
            Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
            to solve the least squares problem. "qr" uses the QR
            factorization.
        cov_type : str, optional
            See `regression.linear_model.RegressionResults` for a description
            of the available covariance estimators.
        cov_kwds : list or None, optional
            See `linear_model.RegressionResults.get_robustcov_results` for a
            description required keywords for alternative covariance
            estimators.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.  Default behavior depends on cov_type. See
            `linear_model.RegressionResults.get_robustcov_results` for
            implementation details.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.

        Returns
        -------
        RegressionResults
            The model estimation results.

        See Also
        --------
        RegressionResults
            The results container.
        RegressionResults.get_robustcov_results
            A method to change the covariance estimator used when fitting the
            model.

        Notes
        -----
        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.
        """
        pass

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model.
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        array_like
            An array of fitted values.

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        pass

    def get_distribution(self, params, scale, exog=None, dist_class=None):
        """
        Construct a random number generator for the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters (regression coefficients).
        scale : scalar
            The variance parameter.
        exog : array_like
            The predictor variable matrix.
        dist_class : class
            A random number generator class.  Must take 'loc' and 'scale'
            as arguments and return a random number generator implementing
            an ``rvs`` method for simulating random values. Defaults to normal.

        Returns
        -------
        gen
            Frozen random number generator object with mean and variance
            determined by the fitted linear model.  Use the ``rvs`` method
            to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``,
        the returned random number generator must be called with
        ``gen.rvs(n)`` where ``n`` is the number of observations in
        the data set used to fit the model.  If any other value is
        used for ``n``, misleading results will be produced.
        """
        pass

class GLS(RegressionModel):
    __doc__ = '\n    Generalized Least Squares\n\n    %(params)s\n    sigma : scalar or array\n        The array or scalar `sigma` is the weighting matrix of the covariance.\n        The default is None for no scaling.  If `sigma` is a scalar, it is\n        assumed that `sigma` is an n x n diagonal matrix with the given\n        scalar, `sigma` as the value of each diagonal element.  If `sigma`\n        is an n-length vector, then `sigma` is assumed to be a diagonal\n        matrix with the given `sigma` on the diagonal.  This should be the\n        same as WLS.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    pinv_wexog : ndarray\n        `pinv_wexog` is the p x n Moore-Penrose pseudoinverse of `wexog`.\n    cholsimgainv : ndarray\n        The transpose of the Cholesky decomposition of the pseudoinverse.\n    df_model : float\n        p - 1, where p is the number of regressors including the intercept.\n        of freedom.\n    df_resid : float\n        Number of observations n less the number of parameters p.\n    llf : float\n        The value of the likelihood function of the fitted model.\n    nobs : float\n        The number of observations n.\n    normalized_cov_params : ndarray\n        p x p array :math:`(X^{T}\\Sigma^{-1}X)^{-1}`\n    results : RegressionResults instance\n        A property that returns the RegressionResults class if fit.\n    sigma : ndarray\n        `sigma` is the n x n covariance structure of the error terms.\n    wexog : ndarray\n        Design matrix whitened by `cholsigmainv`\n    wendog : ndarray\n        Response variable whitened by `cholsigmainv`\n\n    See Also\n    --------\n    WLS : Fit a linear model using Weighted Least Squares.\n    OLS : Fit a linear model using Ordinary Least Squares.\n\n    Notes\n    -----\n    If sigma is a function of the data making one of the regressors\n    a constant, then the current postestimation statistics will not be correct.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.longley.load()\n    >>> data.exog = sm.add_constant(data.exog)\n    >>> ols_resid = sm.OLS(data.endog, data.exog).fit().resid\n    >>> res_fit = sm.OLS(ols_resid[1:], ols_resid[:-1]).fit()\n    >>> rho = res_fit.params\n\n    `rho` is a consistent estimator of the correlation of the residuals from\n    an OLS fit of the longley data.  It is assumed that this is the true rho\n    of the AR process data.\n\n    >>> from scipy.linalg import toeplitz\n    >>> order = toeplitz(np.arange(16))\n    >>> sigma = rho**order\n\n    `sigma` is an n x n matrix of the autocorrelation structure of the\n    data.\n\n    >>> gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)\n    >>> gls_results = gls_model.fit()\n    >>> print(gls_results.summary())\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, sigma=None, missing='none', hasconst=None, **kwargs):
        if type(self) is GLS:
            self._check_kwargs(kwargs)
        sigma, cholsigmainv = _get_sigma(sigma, len(endog))
        super(GLS, self).__init__(endog, exog, missing=missing, hasconst=hasconst, sigma=sigma, cholsigmainv=cholsigmainv, **kwargs)
        self._data_attr.extend(['sigma', 'cholsigmainv'])

    def whiten(self, x):
        """
        GLS whiten method.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        ndarray
            The value np.dot(cholsigmainv,X).

        See Also
        --------
        GLS : Fit a linear model using Generalized Least Squares.
        """
        pass

    def loglike(self, params):
        """
        Compute the value of the Gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.

        Parameters
        ----------
        params : array_like
            The model parameters.

        Returns
        -------
        float
            The value of the log-likelihood function for a GLS Model.

        Notes
        -----
        The log-likelihood function for the normal distribution is

        .. math:: -\\frac{n}{2}\\log\\left(\\left(Y-\\hat{Y}\\right)^{\\prime}
                   \\left(Y-\\hat{Y}\\right)\\right)
                  -\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)
                  -\\frac{1}{2}\\log\\left(\\left|\\Sigma\\right|\\right)

        Y and Y-hat are whitened.
        """
        pass

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute weights for calculating Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        pass

class WLS(RegressionModel):
    __doc__ = '\n    Weighted Least Squares\n\n    The weights are presumed to be (proportional to) the inverse of\n    the variance of the observations.  That is, if the variables are\n    to be transformed by 1/sqrt(W) you must supply weights = 1/W.\n\n    %(params)s\n    weights : array_like, optional\n        A 1d array of weights.  If you supply 1/W then the variables are\n        pre- multiplied by 1/sqrt(W).  If no weights are supplied the\n        default value is 1 and WLS results are the same as OLS.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    weights : ndarray\n        The stored weights supplied as an argument.\n\n    See Also\n    --------\n    GLS : Fit a linear model using Generalized Least Squares.\n    OLS : Fit a linear model using Ordinary Least Squares.\n\n    Notes\n    -----\n    If the weights are a function of the data, then the post estimation\n    statistics such as fvalue and mse_model might not be correct, as the\n    package does not yet support no-constant regression.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> Y = [1,3,4,5,2,3,4]\n    >>> X = range(1,8)\n    >>> X = sm.add_constant(X)\n    >>> wls_model = sm.WLS(Y,X, weights=list(range(1,8)))\n    >>> results = wls_model.fit()\n    >>> results.params\n    array([ 2.91666667,  0.0952381 ])\n    >>> results.tvalues\n    array([ 2.0652652 ,  0.35684428])\n    >>> print(results.t_test([1, 0]))\n    <T test: effect=array([ 2.91666667]), sd=array([[ 1.41224801]]),\n     t=array([[ 2.0652652]]), p=array([[ 0.04690139]]), df_denom=5>\n    >>> print(results.f_test([0, 1]))\n    <F test: F=array([[ 0.12733784]]), p=[[ 0.73577409]], df_denom=5, df_num=1>\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog, weights=1.0, missing='none', hasconst=None, **kwargs):
        if type(self) is WLS:
            self._check_kwargs(kwargs)
        weights = np.array(weights)
        if weights.shape == ():
            if missing == 'drop' and 'missing_idx' in kwargs and (kwargs['missing_idx'] is not None):
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super(WLS, self).__init__(endog, exog, missing=missing, weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, x):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights).

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The whitened values sqrt(weights)*X.
        """
        pass

    def loglike(self, params):
        """
        Compute the value of the gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array_like
            The parameter estimates.

        Returns
        -------
        float
            The value of the log-likelihood function for a WLS Model.

        Notes
        -----
        .. math:: -\\frac{n}{2}\\log SSR
                  -\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)
                  +\\frac{1}{2}\\log\\left(\\left|W\\right|\\right)

        where :math:`W` is a diagonal weight matrix,
        :math:`\\left|W\\right|` is its determinant, and
        :math:`SSR=\\left(Y-\\hat{Y}\\right)^\\prime W \\left(Y-\\hat{Y}\\right)` is
        the sum of the squared weighted residuals.
        """
        pass

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute the weights for calculating the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        pass

class OLS(WLS):
    __doc__ = '\n    Ordinary Least Squares\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    weights : scalar\n        Has an attribute weights = array(1.0) due to inheritance from WLS.\n\n    See Also\n    --------\n    WLS : Fit a linear model using Weighted Least Squares.\n    GLS : Fit a linear model using Generalized Least Squares.\n\n    Notes\n    -----\n    No constant is added by the model unless you are using formulas.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import numpy as np\n    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")\n    >>> Y = duncan_prestige.data[\'income\']\n    >>> X = duncan_prestige.data[\'education\']\n    >>> X = sm.add_constant(X)\n    >>> model = sm.OLS(Y,X)\n    >>> results = model.fit()\n    >>> results.params\n    const        10.603498\n    education     0.594859\n    dtype: float64\n\n    >>> results.tvalues\n    const        2.039813\n    education    6.892802\n    dtype: float64\n\n    >>> print(results.t_test([1, 0]))\n                                 Test for Constraints\n    ==============================================================================\n                     coef    std err          t      P>|t|      [0.025      0.975]\n    ------------------------------------------------------------------------------\n    c0            10.6035      5.198      2.040      0.048       0.120      21.087\n    ==============================================================================\n\n    >>> print(results.f_test(np.identity(2)))\n    <F test: F=array([[159.63031026]]), p=1.2607168903696672e-20,\n     df_denom=43, df_num=2>\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog=None, missing='none', hasconst=None, **kwargs):
        if 'weights' in kwargs:
            msg = 'Weights are not supported in OLS and will be ignoredAn exception will be raised in the next version.'
            warnings.warn(msg, ValueWarning)
        super(OLS, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)
        if 'weights' in self._init_keys:
            self._init_keys.remove('weights')
        if type(self) is OLS:
            self._check_kwargs(kwargs, ['offset'])

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        float
            The likelihood function evaluated at params.
        """
        pass

    def whiten(self, x):
        """
        OLS model whitener does nothing.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The input array unmodified.

        See Also
        --------
        OLS : Fit a linear model using Ordinary Least Squares.
        """
        pass

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The score vector.
        """
        pass

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The Hessian matrix.
        """
        pass

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Calculate the weights for the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        pass

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """
        pass

class GLSAR(GLS):
    __doc__ = '\n    Generalized Least Squares with AR covariance structure\n\n    %(params)s\n    rho : int\n        The order of the autoregressive covariance.\n    %(extra_params)s\n\n    Notes\n    -----\n    GLSAR is considered to be experimental.\n    The linear autoregressive process of order p--AR(p)--is defined as:\n    TODO\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> X = range(1,8)\n    >>> X = sm.add_constant(X)\n    >>> Y = [1,3,4,5,8,10,9]\n    >>> model = sm.GLSAR(Y, X, rho=2)\n    >>> for i in range(6):\n    ...     results = model.fit()\n    ...     print("AR coefficients: {0}".format(model.rho))\n    ...     rho, sigma = sm.regression.yule_walker(results.resid,\n    ...                                            order=model.order)\n    ...     model = sm.GLSAR(Y, X, rho)\n    ...\n    AR coefficients: [ 0.  0.]\n    AR coefficients: [-0.52571491 -0.84496178]\n    AR coefficients: [-0.6104153  -0.86656458]\n    AR coefficients: [-0.60439494 -0.857867  ]\n    AR coefficients: [-0.6048218  -0.85846157]\n    AR coefficients: [-0.60479146 -0.85841922]\n    >>> results.params\n    array([-0.66661205,  1.60850853])\n    >>> results.tvalues\n    array([ -2.10304127,  21.8047269 ])\n    >>> print(results.t_test([1, 0]))\n    <T test: effect=array([-0.66661205]), sd=array([[ 0.31697526]]),\n     t=array([[-2.10304127]]), p=array([[ 0.06309969]]), df_denom=3>\n    >>> print(results.f_test(np.identity(2)))\n    <F test: F=array([[ 1815.23061844]]), p=[[ 0.00002372]],\n     df_denom=3, df_num=2>\n\n    Or, equivalently\n\n    >>> model2 = sm.GLSAR(Y, X, rho=2)\n    >>> res = model2.iterative_fit(maxiter=6)\n    >>> model2.rho\n    array([-0.60479146, -0.85841922])\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + base._extra_param_doc}

    def __init__(self, endog, exog=None, rho=1, missing='none', hasconst=None, **kwargs):
        if isinstance(rho, (int, np.integer)):
            self.order = int(rho)
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0, 1]:
                raise ValueError('AR parameters must be a scalar or a vector')
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            super(GLSAR, self).__init__(endog, np.ones((endog.shape[0], 1)), missing=missing, hasconst=None, **kwargs)
        else:
            super(GLSAR, self).__init__(endog, exog, missing=missing, **kwargs)

    def iterative_fit(self, maxiter=3, rtol=0.0001, **kwargs):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated iteratively.

        Parameters
        ----------
        maxiter : int, optional
            The number of iterations.
        rtol : float, optional
            Relative tolerance between estimated coefficients to stop the
            estimation.  Stops if max(abs(last - current) / abs(last)) < rtol.
        **kwargs
            Additional keyword arguments passed to `fit`.

        Returns
        -------
        RegressionResults
            The results computed using an iterative fit.
        """
        pass

    def whiten(self, x):
        """
        Whiten a series of columns according to an AR(p) covariance structure.

        Whitening using this method drops the initial p observations.

        Parameters
        ----------
        x : array_like
            The data to be whitened.

        Returns
        -------
        ndarray
            The whitened data.
        """
        pass

def yule_walker(x, order=1, method='adjusted', df=None, inv=False, demean=True):
    """
    Estimate AR(p) parameters from a sequence using the Yule-Walker equations.

    Adjusted or maximum-likelihood estimator (mle)

    Parameters
    ----------
    x : array_like
        A 1d array.
    order : int, optional
        The order of the autoregressive process.  Default is 1.
    method : str, optional
       Method can be 'adjusted' or 'mle' and this determines
       denominator in estimate of autocorrelation function (ACF) at
       lag k. If 'mle', the denominator is n=X.shape[0], if 'adjusted'
       the denominator is n-k.  The default is adjusted.
    df : int, optional
       Specifies the degrees of freedom. If `df` is supplied, then it
       is assumed the X has `df` degrees of freedom rather than `n`.
       Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is
        False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho : ndarray
        AR(p) coefficients computed using the Yule-Walker method.
    sigma : float
        The estimate of the residual standard deviation.

    See Also
    --------
    burg : Burg's AR estimator.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Autoregressive_moving_average_model for
    further details.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma = sm.regression.yule_walker(data.endog, order=4,
    ...                                        method="mle")

    >>> rho
    array([ 1.28310031, -0.45240924, -0.20770299,  0.04794365])
    >>> sigma
    16.808022730464351
    """
    pass

def burg(endog, order=1, demean=True):
    """
    Compute Burg's AP(p) parameter estimator.

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    order : int, optional
        Order of the AR.  Default is 1.
    demean : bool, optional
        Flag indicating to subtract the mean from endog before estimation.

    Returns
    -------
    rho : ndarray
        The AR(p) coefficients computed using Burg's algorithm.
    sigma2 : float
        The estimate of the residual variance.

    See Also
    --------
    yule_walker : Estimate AR parameters using the Yule-Walker method.

    Notes
    -----
    AR model estimated includes a constant that is estimated using the sample
    mean (see [1]_). This value is not reported.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.datasets.sunspots import load
    >>> data = load()
    >>> rho, sigma2 = sm.regression.linear_model.burg(data.endog, order=4)

    >>> rho
    array([ 1.30934186, -0.48086633, -0.20185982,  0.05501941])
    >>> sigma2
    271.2467306963966
    """
    pass

class RegressionResults(base.LikelihoodModelResults):
    """
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.

    Attributes
    ----------
    pinv_wexog
        See model class docstring for implementation details.
    cov_type
        Parameter covariance estimator used for standard errors and t-stats.
    df_model
        Model degrees of freedom. The number of regressors `p`. Does not
        include the constant if one is present.
    df_resid
        Residual degrees of freedom. `n - p - 1`, if a constant is present.
        `n - p` if a constant is not included.
    het_scale
        adjusted squared residuals for heteroscedasticity robust standard
        errors. Is only available after `HC#_se` or `cov_HC#` is called.
        See HC#_se for more information.
    history
        Estimation history for iterative estimators.
    model
        A pointer to the model instance that called fit() or results.
    params
        The linear coefficients that minimize the least squares
        criterion.  This is usually called Beta for the classical
        linear model.
    """
    _cache = {}

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        super(RegressionResults, self).__init__(model, params, normalized_cov_params, scale)
        self._cache = {}
        if hasattr(model, 'wexog_singular_values'):
            self._wexog_singular_values = model.wexog_singular_values
        else:
            self._wexog_singular_values = None
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
            if use_t is None:
                use_t = True
            self.use_t = use_t
        else:
            if cov_kwds is None:
                cov_kwds = {}
            if 'use_t' in cov_kwds:
                use_t_2 = cov_kwds.pop('use_t')
                if use_t is None:
                    use_t = use_t_2
            self.get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def conf_int(self, alpha=0.05, cols=None):
        """
        Compute the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Columns to include in returned confidence intervals.

        Returns
        -------
        array_like
            The confidence intervals.

        Notes
        -----
        The confidence interval is based on Student's t-distribution.
        """
        pass

    @cache_readonly
    def nobs(self):
        """Number of observations n."""
        pass

    @cache_readonly
    def fittedvalues(self):
        """The predicted values for the original (unwhitened) design."""
        pass

    @cache_readonly
    def wresid(self):
        """
        The residuals of the transformed/whitened regressand and regressor(s).
        """
        pass

    @cache_readonly
    def resid(self):
        """The residuals of the model."""
        pass

    @cache_writable()
    def scale(self):
        """
        A scale factor for the covariance matrix.

        The Default value is ssr/(n-p).  Note that the square root of `scale`
        is often called the standard error of the regression.
        """
        pass

    @cache_readonly
    def ssr(self):
        """Sum of squared (whitened) residuals."""
        pass

    @cache_readonly
    def centered_tss(self):
        """The total (weighted) sum of squares centered about the mean."""
        pass

    @cache_readonly
    def uncentered_tss(self):
        """
        Uncentered sum of squares.

        The sum of the squared values of the (whitened) endogenous response
        variable.
        """
        pass

    @cache_readonly
    def ess(self):
        """
        The explained sum of squares.

        If a constant is present, the centered total sum of squares minus the
        sum of squared residuals. If there is no constant, the uncentered total
        sum of squares is used.
        """
        pass

    @cache_readonly
    def rsquared(self):
        """
        R-squared of the model.

        This is defined here as 1 - `ssr`/`centered_tss` if the constant is
        included in the model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
        """
        pass

    @cache_readonly
    def rsquared_adj(self):
        """
        Adjusted R-squared.

        This is defined here as 1 - (`nobs`-1)/`df_resid` * (1-`rsquared`)
        if a constant is included and 1 - `nobs`/`df_resid` * (1-`rsquared`) if
        no constant is included.
        """
        pass

    @cache_readonly
    def mse_model(self):
        """
        Mean squared error the model.

        The explained sum of squares divided by the model degrees of freedom.
        """
        pass

    @cache_readonly
    def mse_resid(self):
        """
        Mean squared error of the residuals.

        The sum of squared residuals divided by the residual degrees of
        freedom.
        """
        pass

    @cache_readonly
    def mse_total(self):
        """
        Total mean squared error.

        The uncentered total sum of squares divided by the number of
        observations.
        """
        pass

    @cache_readonly
    def fvalue(self):
        """
        F-statistic of the fully specified model.

        Calculated as the mean squared error of the model divided by the mean
        squared error of the residuals if the nonrobust covariance is used.
        Otherwise computed using a Wald-like quadratic form that tests whether
        all coefficients (excluding the constant) are zero.
        """
        pass

    @cache_readonly
    def f_pvalue(self):
        """The p-value of the F-statistic."""
        pass

    @cache_readonly
    def bse(self):
        """The standard errors of the parameter estimates."""
        pass

    @cache_readonly
    def aic(self):
        """
        Akaike's information criteria.

        For a model with a constant :math:`-2llf + 2(df\\_model + 1)`. For a
        model without a constant :math:`-2llf + 2(df\\_model)`.
        """
        pass

    @cache_readonly
    def bic(self):
        """
        Bayes' information criteria.

        For a model with a constant :math:`-2llf + \\log(n)(df\\_model+1)`.
        For a model without a constant :math:`-2llf + \\log(n)(df\\_model)`.
        """
        pass

    def info_criteria(self, crit, dk_params=0):
        """Return an information criterion for the model.

        Parameters
        ----------
        crit : string
            One of 'aic', 'bic', 'aicc' or 'hqic'.
        dk_params : int or float
            Correction to the number of parameters used in the information
            criterion. By default, only mean parameters are included, the
            scale parameter is not included in the parameter count.
            Use ``dk_params=1`` to include scale in the parameter count.

        Returns
        -------
        Value of information criterion.

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        pass

    @cache_readonly
    def eigenvals(self):
        """
        Return eigenvalues sorted in decreasing order.
        """
        pass

    @cache_readonly
    def condition_number(self):
        """
        Return condition number of exogenous matrix.

        Calculated as ratio of largest to smallest singular value of the
        exogenous variables. This value is the same as the square root of
        the ratio of the largest to smallest eigenvalue of the inner-product
        of the exogenous variables.
        """
        pass

    @cache_readonly
    def cov_HC0(self):
        """
        Heteroscedasticity robust covariance matrix. See HC0_se.
        """
        pass

    @cache_readonly
    def cov_HC1(self):
        """
        Heteroscedasticity robust covariance matrix. See HC1_se.
        """
        pass

    @cache_readonly
    def cov_HC2(self):
        """
        Heteroscedasticity robust covariance matrix. See HC2_se.
        """
        pass

    @cache_readonly
    def cov_HC3(self):
        """
        Heteroscedasticity robust covariance matrix. See HC3_se.
        """
        pass

    @cache_readonly
    def HC0_se(self):
        """
        White's (1980) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i].

        When HC0_se or cov_HC0 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is just
        resid**2.
        """
        pass

    @cache_readonly
    def HC1_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as sqrt(diag(n/(n-p)*HC_0).

        When HC1_se or cov_HC1 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        n/(n-p)*resid**2.
        """
        pass

    @cache_readonly
    def HC2_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T

        When HC2_se or cov_HC2 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii).
        """
        pass

    @cache_readonly
    def HC3_se(self):
        """
        MacKinnon and White's (1985) heteroskedasticity robust standard errors.

        Notes
        -----
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T.

        When HC3_se or cov_HC3 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii)^(2).
        """
        pass

    @cache_readonly
    def resid_pearson(self):
        """
        Residuals, normalized to have unit variance.

        Returns
        -------
        array_like
            The array `wresid` normalized by the sqrt of the scale to have
            unit variance.
        """
        pass

    def _is_nested(self, restricted):
        """
        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current
            model. The result instance of the restricted model is required to
            have two attributes, residual sum of squares, `ssr`, residual
            degrees of freedom, `df_resid`.

        Returns
        -------
        nested : bool
            True if nested, otherwise false

        Notes
        -----
        A most nests another model if the regressors in the smaller
        model are spanned by the regressors in the larger model and
        the regressand is identical.
        """
        pass

    def compare_lm_test(self, restricted, demean=True, use_lr=False):
        """
        Use Lagrange Multiplier test to test a set of linear restrictions.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.
        demean : bool
            Flag indicating whether the demean the scores based on the
            residuals from the restricted model.  If True, the covariance of
            the scores are used and the LM test is identical to the large
            sample version of the LR test.
        use_lr : bool
            A flag indicating whether to estimate the covariance of the model
            scores using the unrestricted model. Setting the to True improves
            the power of the test.

        Returns
        -------
        lm_value : float
            The test statistic which has a chi2 distributed.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The LM test examines whether the scores from the restricted model are
        0. If the null is true, and the restrictions are valid, then the
        parameters of the restricted model should be close to the minimum of
        the sum of squared errors, and so the scores should be close to zero,
        on average.
        """
        pass

    def compare_f_test(self, restricted):
        """
        Use F test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.

        Returns
        -------
        f_value : float
            The test statistic which has an F distribution.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in
            df between models.

        Notes
        -----
        See mailing list discussion October 17,

        This test compares the residual sum of squares of the two
        models.  This is not a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results under
        the assumption of homoscedasticity and no autocorrelation
        (sphericity).
        """
        pass

    def compare_lr_test(self, restricted, large_sample=False):
        """
        Likelihood ratio test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current model.
            The result instance of the restricted model is required to have two
            attributes, residual sum of squares, `ssr`, residual degrees of
            freedom, `df_resid`.

        large_sample : bool
            Flag indicating whether to use a heteroskedasticity robust version
            of the LR test, which is a modified LM test.

        Returns
        -------
        lr_stat : float
            The likelihood ratio which is chisquare distributed with df_diff
            degrees of freedom.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The exact likelihood ratio is valid for homoskedastic data,
        and is defined as

        .. math:: D=-2\\log\\left(\\frac{\\mathcal{L}_{null}}
           {\\mathcal{L}_{alternative}}\\right)

        where :math:`\\mathcal{L}` is the likelihood of the
        model. With :math:`D` distributed as chisquare with df equal
        to difference in number of parameters or equivalently
        difference in residual degrees of freedom.

        The large sample version of the likelihood ratio is defined as

        .. math:: D=n s^{\\prime}S^{-1}s

        where :math:`s=n^{-1}\\sum_{i=1}^{n} s_{i}`

        .. math:: s_{i} = x_{i,alternative} \\epsilon_{i,null}

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.
        """
        pass

    def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwargs):
        """
        Create new results instance with robust covariance as default.

        Parameters
        ----------
        cov_type : str
            The type of robust sandwich estimator to use. See Notes below.
        use_t : bool
            If true, then the t distribution is used for inference.
            If false, then the normal distribution is used.
            If `use_t` is None, then an appropriate default is used, which is
            `True` if the cov_type is nonrobust, and `False` in all other
            cases.
        **kwargs
            Required or optional arguments for robust covariance calculation.
            See Notes below.

        Returns
        -------
        RegressionResults
            This method creates a new results instance with the
            requested robust covariance as the default covariance of
            the parameters.  Inferential statistics like p-values and
            hypothesis tests will be based on this covariance matrix.

        Notes
        -----
        The following covariance types and required or optional arguments are
        currently available:

        - 'fixed scale' uses a predefined scale

          ``scale``: float, optional
            Argument to set the scale. Default is 1.

        - 'HC0', 'HC1', 'HC2', 'HC3': heteroscedasticity robust covariance

          - no keyword arguments

        - 'HAC': heteroskedasticity-autocorrelation robust covariance

          ``maxlags`` :  integer, required
            number of lags to use

          ``kernel`` : {callable, str}, optional
            kernels currently available kernels are ['bartlett', 'uniform'],
            default is Bartlett

          ``use_correction``: bool, optional
            If true, use small sample correction

        - 'cluster': clustered covariance estimator

          ``groups`` : array_like[int], required :
            Integer-valued index of clusters or groups.

          ``use_correction``: bool, optional
            If True the sandwich covariance is calculated with a small
            sample correction.
            If False the sandwich covariance is calculated without
            small sample correction.

          ``df_correction``: bool, optional
            If True (default), then the degrees of freedom for the
            inferential statistics and hypothesis tests, such as
            pvalues, f_pvalue, conf_int, and t_test and f_test, are
            based on the number of groups minus one instead of the
            total number of observations minus the number of explanatory
            variables. `df_resid` of the results instance is also
            adjusted. When `use_t` is also True, then pvalues are
            computed using the Student's t distribution using the
            corrected values. These may differ substantially from
            p-values based on the normal is the number of groups is
            small.
            If False, then `df_resid` of the results instance is not
            adjusted.

        - 'hac-groupsum': Driscoll and Kraay, heteroscedasticity and
          autocorrelation robust covariance for panel data
          # TODO: more options needed here

          ``time`` : array_like, required
            index of time periods
          ``maxlags`` : integer, required
            number of lags to use
          ``kernel`` : {callable, str}, optional
            The available kernels are ['bartlett', 'uniform']. The default is
            Bartlett.
          ``use_correction`` : {False, 'hac', 'cluster'}, optional
            If False the the sandwich covariance is calculated without small
            sample correction. If `use_correction = 'cluster'` (default),
            then the same small sample correction as in the case of
            `covtype='cluster'` is used.
          ``df_correction`` : bool, optional
            The adjustment to df_resid, see cov_type 'cluster' above

        - 'hac-panel': heteroscedasticity and autocorrelation robust standard
          errors in panel data. The data needs to be sorted in this case, the
          time series for each panel unit or cluster need to be stacked. The
          membership to a time series of an individual or group can be either
          specified by group indicators or by increasing time periods. One of
          ``groups`` or ``time`` is required. # TODO: we need more options here

          ``groups`` : array_like[int]
            indicator for groups
          ``time`` : array_like[int]
            index of time periods
          ``maxlags`` : int, required
            number of lags to use
          ``kernel`` : {callable, str}, optional
            Available kernels are ['bartlett', 'uniform'], default
            is Bartlett
          ``use_correction`` : {False, 'hac', 'cluster'}, optional
            If False the sandwich covariance is calculated without
            small sample correction.
          ``df_correction`` : bool, optional
            Adjustment to df_resid, see cov_type 'cluster' above

        **Reminder**: ``use_correction`` in "hac-groupsum" and "hac-panel" is
        not bool, needs to be in {False, 'hac', 'cluster'}.

        .. todo:: Currently there is no check for extra or misspelled keywords,
             except in the case of cov_type `HCx`
        """
        pass

    def summary(self, yname: str | None=None, xname: Sequence[str] | None=None, title: str | None=None, alpha: float=0.05, slim: bool=False):
        """
        Summarize the Regression Results.

        Parameters
        ----------
        yname : str, optional
            Name of endogenous (response) variable. The Default is `y`.
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float, optional
            The significance level for the confidence intervals.
        slim : bool, optional
            Flag indicating to produce reduced set or diagnostic information.
            Default is False.

        Returns
        -------
        Summary
            Instance holding the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : A class that holds summary results.
        """
        pass

    def summary2(self, yname: str | None=None, xname: Sequence[str] | None=None, title: str | None=None, alpha: float=0.05, float_format: str='%.4f'):
        """
        Experimental summary function to summarize the regression results.

        Parameters
        ----------
        yname : str
            The name of the dependent variable (optional).
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.
        float_format : str
            The format for floats in parameters summary.

        Returns
        -------
        Summary
            Instance holding the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary
            A class that holds summary results.
        """
        pass

class OLSResults(RegressionResults):
    """
    Results class for for an OLS model.

    Parameters
    ----------
    model : RegressionModel
        The regression model instance.
    params : ndarray
        The estimated parameters.
    normalized_cov_params : ndarray
        The normalized covariance parameters.
    scale : float
        The estimated scale of the residuals.
    cov_type : str
        The covariance estimator used in the results.
    cov_kwds : dict
        Additional keywords used in the covariance specification.
    use_t : bool
        Flag indicating to use the Student's t in inference.
    **kwargs
        Additional keyword arguments used to initialize the results.

    See Also
    --------
    RegressionResults
        Results store for WLS and GLW models.

    Notes
    -----
    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:

    - get_influence
    - outlier_test
    - el_test
    - conf_int_el
    """

    def get_influence(self):
        """
        Calculate influence and outlier measures.

        Returns
        -------
        OLSInfluence
            The instance containing methods to calculate the main influence and
            outlier measures for the OLS regression.

        See Also
        --------
        statsmodels.stats.outliers_influence.OLSInfluence
            A class that exposes methods to examine observation influence.
        """
        pass

    def outlier_test(self, method='bonf', alpha=0.05, labels=None, order=False, cutoff=None):
        """
        Test observations for outliers according to method.

        Parameters
        ----------
        method : str
            The method to use in the outlier test.  Must be one of:

            - `bonferroni` : one-step correction
            - `sidak` : one-step correction
            - `holm-sidak` :
            - `holm` :
            - `simes-hochberg` :
            - `hommel` :
            - `fdr_bh` : Benjamini/Hochberg
            - `fdr_by` : Benjamini/Yekutieli

            See `statsmodels.stats.multitest.multipletests` for details.
        alpha : float
            The familywise error rate (FWER).
        labels : None or array_like
            If `labels` is not None, then it will be used as index to the
            returned pandas DataFrame. See also Returns below.
        order : bool
            Whether or not to order the results by the absolute value of the
            studentized residuals. If labels are provided they will also be
            sorted.
        cutoff : None or float in [0, 1]
            If cutoff is not None, then the return only includes observations
            with multiple testing corrected p-values strictly below the cutoff.
            The returned array or dataframe can be empty if t.

        Returns
        -------
        array_like
            Returns either an ndarray or a DataFrame if labels is not None.
            Will attempt to get labels from model_results if available. The
            columns are the Studentized residuals, the unadjusted p-value,
            and the corrected p-value according to method.

        Notes
        -----
        The unadjusted p-value is stats.t.sf(abs(resid), df) where
        df = df_resid - 1.
        """
        pass

    def el_test(self, b0_vals, param_nums, return_weights=0, ret_params=0, method='nm', stochastic_exog=1):
        """
        Test single or joint hypotheses using Empirical Likelihood.

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value of the parameter to be tested.
        param_nums : 1darray
            The parameter number to be tested.
        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals. The default is False.
        ret_params : bool
            If true, returns the parameter vector that maximizes the likelihood
            ratio at b0_vals.  Also returns the weights.  The default is False.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors. The default is True.

        Returns
        -------
        tuple
            The p-value and -2 times the log-likelihood ratio for the
            hypothesized values.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.stackloss.load()
        >>> endog = data.endog
        >>> exog = sm.add_constant(data.exog)
        >>> model = sm.OLS(endog, exog)
        >>> fitted = model.fit()
        >>> fitted.params
        >>> array([-39.91967442,   0.7156402 ,   1.29528612,  -0.15212252])
        >>> fitted.rsquared
        >>> 0.91357690446068196
        >>> # Test that the slope on the first variable is 0
        >>> fitted.el_test([0], [1])
        >>> (27.248146353888796, 1.7894660442330235e-07)
        """
        pass

    def conf_int_el(self, param_num, sig=0.05, upper_bound=None, lower_bound=None, method='nm', stochastic_exog=True):
        """
        Compute the confidence interval using Empirical Likelihood.

        Parameters
        ----------
        param_num : float
            The parameter for which the confidence interval is desired.
        sig : float
            The significance level.  Default is 0.05.
        upper_bound : float
            The maximum value the upper limit can be.  Default is the
            99.9% confidence value under OLS assumptions.
        lower_bound : float
            The minimum value the lower limit can be.  Default is the 99.9%
            confidence value under OLS assumptions.
        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            The default is 'nm'.
        stochastic_exog : bool
            When True, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  The default is True.

        Returns
        -------
        lowerl : float
            The lower bound of the confidence interval.
        upperl : float
            The upper bound of the confidence interval.

        See Also
        --------
        el_test : Test parameters using Empirical Likelihood.

        Notes
        -----
        This function uses brentq to find the value of beta where
        test_beta([beta], param_num)[1] is equal to the critical value.

        The function returns the results of each iteration of brentq at each
        value of beta.

        The current function value of the last printed optimization should be
        the critical value at the desired significance level. For alpha=.05,
        the value is 3.841459.

        To ensure optimization terminated successfully, it is suggested to do
        el_test([lower_limit], [param_num]).

        If the optimization does not terminate successfully, consider switching
        optimization algorithms.

        If optimization is still not successful, try changing the values of
        start_int_params.  If the current function value repeatedly jumps
        from a number between 0 and the critical value and a very large number
        (>50), the starting parameters of the interior minimization need
        to be changed.
        """
        pass

class RegressionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'chisq': 'columns', 'sresid': 'rows', 'weights': 'rows', 'wresid': 'rows', 'bcov_unscaled': 'cov', 'bcov_scaled': 'cov', 'HC0_se': 'columns', 'HC1_se': 'columns', 'HC2_se': 'columns', 'HC3_se': 'columns', 'norm_resid': 'rows'}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(RegressionResultsWrapper, RegressionResults)