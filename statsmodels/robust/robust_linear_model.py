"""
Robust linear models with support for the M-estimators  listed under
:ref:`norms <norms>`.

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York.  1981.

PJ Huber.  1973,  'The 1972 Wald Memorial Lectures: Robust Regression:
    Asymptotics, Conjectures, and Monte Carlo.'  The Annals of Statistics,
    1.5, 799-821.

R Venables, B Ripley. 'Modern Applied Statistics in S'  Springer, New York,
    2002.
"""
import numpy as np
import scipy.stats as stats
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
__all__ = ['RLM']

class RLM(base.LikelihoodModel):
    __doc__ = '\n    Robust Linear Model\n\n    Estimate a robust linear model via iteratively reweighted least squares\n    given a robust criterion estimator.\n\n    %(params)s\n    M : statsmodels.robust.norms.RobustNorm, optional\n        The robust criterion function for downweighting outliers.\n        The current options are LeastSquares, HuberT, RamsayE, AndrewWave,\n        TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().\n        See statsmodels.robust.norms for more information.\n    %(extra_params)s\n\n    Attributes\n    ----------\n\n    df_model : float\n        The degrees of freedom of the model.  The number of regressors p less\n        one for the intercept.  Note that the reported model degrees\n        of freedom does not count the intercept as a regressor, though\n        the model is assumed to have an intercept.\n    df_resid : float\n        The residual degrees of freedom.  The number of observations n\n        less the number of regressors p.  Note that here p does include\n        the intercept as using a degree of freedom.\n    endog : ndarray\n        See above.  Note that endog is a reference to the data so that if\n        data is already an array and it is changed, then `endog` changes\n        as well.\n    exog : ndarray\n        See above.  Note that endog is a reference to the data so that if\n        data is already an array and it is changed, then `endog` changes\n        as well.\n    M : statsmodels.robust.norms.RobustNorm\n         See above.  Robust estimator instance instantiated.\n    nobs : float\n        The number of observations n\n    pinv_wexog : ndarray\n        The pseudoinverse of the design / exogenous data array.  Note that\n        RLM has no whiten method, so this is just the pseudo inverse of the\n        design.\n    normalized_cov_params : ndarray\n        The p x p normalized covariance of the design / exogenous data.\n        This is approximately equal to (X.T X)^(-1)\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.stackloss.load()\n    >>> data.exog = sm.add_constant(data.exog)\n    >>> rlm_model = sm.RLM(data.endog, data.exog,                            M=sm.robust.norms.HuberT())\n\n    >>> rlm_results = rlm_model.fit()\n    >>> rlm_results.params\n    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])\n    >>> rlm_results.bse\n    array([ 0.11100521,  0.30293016,  0.12864961,  9.79189854])\n    >>> rlm_results_HC2 = rlm_model.fit(cov="H2")\n    >>> rlm_results_HC2.params\n    array([  0.82938433,   0.92606597,  -0.12784672, -41.02649835])\n    >>> rlm_results_HC2.bse\n    array([ 0.11945975,  0.32235497,  0.11796313,  9.08950419])\n    >>> mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.Hampel())\n    >>> rlm_hamp_hub = mod.fit(scale_est=sm.robust.scale.HuberScale())\n    >>> rlm_hamp_hub.params\n    array([  0.73175452,   1.25082038,  -0.14794399, -40.27122257])\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc}

    def __init__(self, endog, exog, M=None, missing='none', **kwargs):
        self._check_kwargs(kwargs)
        self.M = M if M is not None else norms.HuberT()
        super(base.LikelihoodModel, self).__init__(endog, exog, missing=missing, **kwargs)
        self._initialize()
        self._data_attr.extend(['weights', 'pinv_wexog'])

    def _initialize(self):
        """
        Initializes the model for the IRLS fit.

        Resets the history and number of iterations.
        """
        pass

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model
        exog : array_like, optional.
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        An array of fitted values
        """
        pass

    def deviance(self, tmp_results):
        """
        Returns the (unnormalized) log-likelihood from the M estimator.
        """
        pass

    def _estimate_scale(self, resid):
        """
        Estimates the scale based on the option provided to the fit method.
        """
        pass

    def fit(self, maxiter=50, tol=1e-08, scale_est='mad', init=None, cov='H1', update_scale=True, conv='dev', start_params=None):
        """
        Fits the model using iteratively reweighted least squares.

        The IRLS routine runs until the specified objective converges to `tol`
        or `maxiter` has been reached.

        Parameters
        ----------
        conv : str
            Indicates the convergence criteria.
            Available options are "coefs" (the coefficients), "weights" (the
            weights in the iteration), "sresid" (the standardized residuals),
            and "dev" (the un-normalized log-likelihood for the M
            estimator).  The default is "dev".
        cov : str, optional
            'H1', 'H2', or 'H3'
            Indicates how the covariance matrix is estimated.  Default is 'H1'.
            See rlm.RLMResults for more information.
        init : str
            Specifies method for the initial estimates of the parameters.
            Default is None, which means that the least squares estimate
            is used.  Currently it is the only available choice.
        maxiter : int
            The maximum number of iterations to try. Default is 50.
        scale_est : str or HuberScale()
            'mad' or HuberScale()
            Indicates the estimate to use for scaling the weights in the IRLS.
            The default is 'mad' (median absolute deviation.  Other options are
            'HuberScale' for Huber's proposal 2. Huber's proposal 2 has
            optional keyword arguments d, tol, and maxiter for specifying the
            tuning constant, the convergence tolerance, and the maximum number
            of iterations. See statsmodels.robust.scale for more information.
        tol : float
            The convergence tolerance of the estimate.  Default is 1e-8.
        update_scale : Bool
            If `update_scale` is False then the scale estimate for the
            weights is held constant over the iteration.  Otherwise, it
            is updated for each fit in the iteration.  Default is True.
        start_params : array_like, optional
            Initial guess of the solution of the optimizer. If not provided,
            the initial parameters are computed using OLS.

        Returns
        -------
        results : statsmodels.rlm.RLMresults
            Results instance
        """
        pass

class RLMResults(base.LikelihoodModelResults):
    """
    Class to contain RLM results

    Attributes
    ----------

    bcov_scaled : ndarray
        p x p scaled covariance matrix specified in the model fit method.
        The default is H1. H1 is defined as
        ``k**2 * (1/df_resid*sum(M.psi(sresid)**2)*scale**2)/
        ((1/nobs*sum(M.psi_deriv(sresid)))**2) * (X.T X)^(-1)``

        where ``k = 1 + (df_model +1)/nobs * var_psiprime/m**2``
        where ``m = mean(M.psi_deriv(sresid))`` and
        ``var_psiprime = var(M.psi_deriv(sresid))``

        H2 is defined as
        ``k * (1/df_resid) * sum(M.psi(sresid)**2) *scale**2/
        ((1/nobs)*sum(M.psi_deriv(sresid)))*W_inv``

        H3 is defined as
        ``1/k * (1/df_resid * sum(M.psi(sresid)**2)*scale**2 *
        (W_inv X.T X W_inv))``

        where `k` is defined as above and
        ``W_inv = (M.psi_deriv(sresid) exog.T exog)^(-1)``

        See the technical documentation for cleaner formulae.
    bcov_unscaled : ndarray
        The usual p x p covariance matrix with scale set equal to 1.  It
        is then just equivalent to normalized_cov_params.
    bse : ndarray
        An array of the standard errors of the parameters.  The standard
        errors are taken from the robust covariance matrix specified in the
        argument to fit.
    chisq : ndarray
        An array of the chi-squared values of the parameter estimates.
    df_model
        See RLM.df_model
    df_resid
        See RLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `deviance`,
        `params`, `iteration` and the convergence criteria specified in
        `RLM.fit`, if different from `deviance` or `params`.
    fit_options : dict
        Contains the options given to fit.
    fittedvalues : ndarray
        The linear predicted values.  dot(exog, params)
    model : statsmodels.rlm.RLM
        A reference to the model instance
    nobs : float
        The number of observations n
    normalized_cov_params : ndarray
        See RLM.normalized_cov_params
    params : ndarray
        The coefficients of the fitted model
    pinv_wexog : ndarray
        See RLM.pinv_wexog
    pvalues : ndarray
        The p values associated with `tvalues`. Note that `tvalues` are assumed
        to be distributed standard normal rather than Student's t.
    resid : ndarray
        The residuals of the fitted model.  endog - fittedvalues
    scale : float
        The type of scale is determined in the arguments to the fit method in
        RLM.  The reported scale is taken from the residuals of the weighted
        least squares in the last IRLS iteration if update_scale is True.  If
        update_scale is False, then it is the scale given by the first OLS
        fit before the IRLS iterations.
    sresid : ndarray
        The scaled residuals.
    tvalues : ndarray
        The "t-statistics" of params. These are defined as params/bse where
        bse are taken from the robust covariance matrix specified in the
        argument to fit.
    weights : ndarray
        The reported weights are determined by passing the scaled residuals
        from the last weighted least squares fit in the IRLS algorithm.

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale):
        super(RLMResults, self).__init__(model, params, normalized_cov_params, scale)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        self._cache = {}
        self._data_in_cache.extend(['sresid'])
        self.cov_params_default = self.bcov_scaled

    def summary(self, yname=None, xname=None, title=0, alpha=0.05, return_fmt='text'):
        """
        This is for testing the new summary setup
        """
        pass

    def summary2(self, xname=None, yname=None, title=None, alpha=0.05, float_format='%.4f'):
        """Experimental summary function for regression results

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        pass

class RLMResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(RLMResultsWrapper, RLMResults)