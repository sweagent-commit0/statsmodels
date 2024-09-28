"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import TimeSeriesModel, TimeSeriesResultsWrapper
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults, NormalityTestResults, WhitenessTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

def ma_rep(coefs, maxn=10):
    """
    MA(\\infty) representation of VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    maxn : int
        Number of MA matrices to compute

    Notes
    -----
    VAR(p) process as

    .. math:: y_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + u_t

    can be equivalently represented as

    .. math:: y_t = \\mu + \\sum_{i=0}^\\infty \\Phi_i u_{t-i}

    e.g. can recursively compute the \\Phi_i matrices with \\Phi_0 = I_k

    Returns
    -------
    phis : ndarray (maxn + 1 x k x k)
    """
    pass

def is_stable(coefs, verbose=False):
    """
    Determine stability of VAR(p) system by examining the eigenvalues of the
    VAR(1) representation

    Parameters
    ----------
    coefs : ndarray (p x k x k)

    Returns
    -------
    is_stable : bool
    """
    pass

def var_acf(coefs, sig_u, nlags=None):
    """
    Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)
    process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        Coefficient matrices A_i
    sig_u : ndarray (k x k)
        Covariance of white noise process u_t
    nlags : int, optional
        Defaults to order p of system

    Notes
    -----
    Ref: Lütkepohl p.28-29

    Returns
    -------
    acf : ndarray, (p, k, k)
    """
    pass

def _var_acf(coefs, sig_u):
    """
    Compute autocovariance function ACF_y(h) for h=1,...,p

    Notes
    -----
    Lütkepohl (2005) p.29
    """
    pass

def forecast_cov(ma_coefs, sigma_u, steps):
    """
    Compute theoretical forecast error variance matrices

    Parameters
    ----------
    steps : int
        Number of steps ahead

    Notes
    -----
    .. math:: \\mathrm{MSE}(h) = \\sum_{i=0}^{h-1} \\Phi \\Sigma_u \\Phi^T

    Returns
    -------
    forc_covs : ndarray (steps x neqs x neqs)
    """
    pass
mse = forecast_cov

def forecast(y, coefs, trend_coefs, steps, exog=None):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
    coefs : ndarray (k_ar x neqs x neqs)
    trend_coefs : ndarray (1 x neqs) or (neqs)
    steps : int
    exog : ndarray (trend_coefs.shape[1] x neqs)

    Returns
    -------
    forecasts : ndarray (steps x neqs)

    Notes
    -----
    Lütkepohl p. 37
    """
    pass

def _forecast_vars(steps, ma_coefs, sig_u):
    """_forecast_vars function used by VECMResults. Note that the definition
    of the local variable covs is the same as in VARProcess and as such it
    differs from the one in VARResults!

    Parameters
    ----------
    steps
    ma_coefs
    sig_u

    Returns
    -------
    """
    pass

def var_loglike(resid, omega, nobs):
    """
    Returns the value of the VAR(p) log-likelihood.

    Parameters
    ----------
    resid : ndarray (T x K)
    omega : ndarray
        Sigma hat matrix.  Each element i,j is the average product of the
        OLS residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/nobs.  There should be no correction for the
        degrees of freedom.
    nobs : int

    Returns
    -------
    llf : float
        The value of the loglikelihood function for a VAR(p) model

    Notes
    -----
    The loglikelihood function for the VAR(p) is

    .. math::

        -\\left(\\frac{T}{2}\\right)
        \\left(\\ln\\left|\\Omega\\right|-K\\ln\\left(2\\pi\\right)-K\\right)
    """
    pass

def orth_ma_rep(results, maxn=10, P=None):
    """Compute Orthogonalized MA coefficient matrices using P matrix such
    that :math:`\\Sigma_u = PP^\\prime`. P defaults to the Cholesky
    decomposition of :math:`\\Sigma_u`

    Parameters
    ----------
    results : VARResults or VECMResults
    maxn : int
        Number of coefficient matrices to compute
    P : ndarray (neqs x neqs), optional
        Matrix such that Sigma_u = PP', defaults to the Cholesky decomposition.

    Returns
    -------
    coefs : ndarray (maxn x neqs x neqs)
    """
    pass

def test_normality(results, signif=0.05):
    """
    Test assumption of normal-distributed errors using Jarque-Bera-style
    omnibus Chi^2 test

    Parameters
    ----------
    results : VARResults or statsmodels.tsa.vecm.vecm.VECMResults
    signif : float
        The test's significance level.

    Notes
    -----
    H0 (null) : data are generated by a Gaussian-distributed process

    Returns
    -------
    result : NormalityTestResults

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
       *Analysis*. Springer.

    .. [2] Kilian, L. & Demiroglu, U. (2000). "Residual-Based Tests for
       Normality in Autoregressions: Asymptotic Theory and Simulation
       Evidence." Journal of Business & Economic Statistics
    """
    pass

class LagOrderResults:
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).
    vecm : bool, default: `False`
        `True` indicates that the model is a VECM. In case of a VAR model
        this argument must be `False`.

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """

    def __init__(self, ics, selected_orders, vecm=False):
        self.title = ('VECM' if vecm else 'VAR') + ' Order Selection'
        self.title += ' (* highlights the minimums)'
        self.ics = ics
        self.selected_orders = selected_orders
        self.vecm = vecm
        self.aic = selected_orders['aic']
        self.bic = selected_orders['bic']
        self.hqic = selected_orders['hqic']
        self.fpe = selected_orders['fpe']

    def __str__(self):
        return f'<{self.__module__}.{self.__class__.__name__} object. Selected orders are: AIC -> {str(self.aic)}, BIC -> {str(self.bic)}, FPE -> {str(self.fpe)}, HQIC ->  {str(self.hqic)}>'

class VAR(TimeSeriesModel):
    """
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable. The independent variable.
    exog : array_like
        2-d exogenous variable.
    dates : array_like
        must match number of rows of endog

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """
    y = deprecated_alias('y', 'endog', remove_version='0.11.0')

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none'):
        super().__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError('Only gave one variable to VAR')
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        """
        Returns in-sample predictions or forecasts
        """
        pass

    def fit(self, maxlags: int | None=None, method='ols', ic=None, trend='c', verbose=False):
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : {int, None}, default None
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend : str {"c", "ct", "ctt", "n"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Returns
        -------
        VARResults
            Estimation results

        Notes
        -----
        See Lütkepohl pp. 146-153 for implementation details.
        """
        pass

    def _estimate_var(self, lags, offset=0, trend='c'):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : {str, None}
            As per above
        """
        pass

    def select_order(self, maxlags=None, trend='c'):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trend : str {"n", "c", "ct", "ctt"}
            * "n" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
        pass

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        """
        Not implemented. Formulas are not supported for VAR models.
        """
        pass

class VARProcess:
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        coefficients for lags of endog, part or params reshaped
    coefs_exog : ndarray
        parameters for trend and user provided exog
    sigma_u : ndarray (k x k)
        residual covariance
    names : sequence (length k)
    _params_info : dict
        internal dict to provide information about the composition of `params`,
        specifically `k_trend` (trend order) and `k_exog_user` (the number of
        exog variables provided by the user).
        If it is None, then coefs_exog are assumed to be for the intercept and
        trend.
    """

    def __init__(self, coefs, coefs_exog, sigma_u, names=None, _params_info=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.coefs_exog = coefs_exog
        self.sigma_u = sigma_u
        self.names = names
        if _params_info is None:
            _params_info = {}
        self.k_exog_user = _params_info.get('k_exog_user', 0)
        if self.coefs_exog is not None:
            k_ex = self.coefs_exog.shape[0] if self.coefs_exog.ndim != 1 else 1
            k_c = k_ex - self.k_exog_user
        else:
            k_c = 0
        self.k_trend = _params_info.get('k_trend', k_c)
        self.k_exog = self.k_trend + self.k_exog_user
        if self.k_trend > 0:
            if coefs_exog.ndim == 2:
                self.intercept = coefs_exog[:, 0]
            else:
                self.intercept = coefs_exog
        else:
            self.intercept = np.zeros(self.neqs)

    def get_eq_index(self, name):
        """Return integer position of requested equation name"""
        pass

    def __str__(self):
        output = 'VAR(%d) process for %d-dimensional response y_t' % (self.k_ar, self.neqs)
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()
        return output

    def is_stable(self, verbose=False):
        """Determine stability based on model coefficients

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of the VAR(1) companion

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues of
        the companion matrix must lie outside the unit circle
        """
        pass

    def simulate_var(self, steps=None, offset=None, seed=None, initial_values=None, nsimulations=None):
        """
        simulate the VAR(p) process for the desired number of steps

        Parameters
        ----------
        steps : None or int
            number of observations to simulate, this includes the initial
            observations to start the autoregressive process.
            If offset is not None, then exog of the model are used if they were
            provided in the model
        offset : None or ndarray (steps, neqs)
            If not None, then offset is added as an observation specific
            intercept to the autoregression. If it is None and either trend
            (including intercept) or exog were used in the VAR model, then
            the linear predictor of those components will be used as offset.
            This should have the same number of rows as steps, and the same
            number of columns as endogenous variables (neqs).
        seed : {None, int}
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random.
        initial_values : array_like, optional
            Initial values for use in the simulation. Shape should be
            (nlags, neqs) or (neqs,). Values should be ordered from less to
            most recent. Note that this values will be returned by the
            simulation as the first values of `endog_simulated` and they
            will count for the total number of steps.
        nsimulations : {None, int}
            Number of simulations to perform. If `nsimulations` is None it will
            perform one simulation and return value will have shape (steps, neqs).

        Returns
        -------
        endog_simulated : nd_array
            Endog of the simulated VAR process. Shape will be (nsimulations, steps, neqs)
            or (steps, neqs) if `nsimulations` is None.
        """
        pass

    def plotsim(self, steps=None, offset=None, seed=None):
        """
        Plot a simulation from the VAR(p) process for the desired number of
        steps
        """
        pass

    def intercept_longrun(self):
        """
        Long run intercept of stable VAR process

        Lütkepohl eq. 2.1.23

        .. math:: \\mu = (I - A_1 - \\dots - A_p)^{-1} \\alpha

        where \\alpha is the intercept (parameter of the constant)
        """
        pass

    def mean(self):
        """
        Long run intercept of stable VAR process

        Warning: trend and exog except for intercept are ignored for this.
        This might change in future versions.

        Lütkepohl eq. 2.1.23

        .. math:: \\mu = (I - A_1 - \\dots - A_p)^{-1} \\alpha

        where \\alpha is the intercept (parameter of the constant)
        """
        pass

    def ma_rep(self, maxn=10):
        """
        Compute MA(:math:`\\infty`) coefficient matrices

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        pass

    def orth_ma_rep(self, maxn=10, P=None):
        """
        Compute orthogonalized MA coefficient matrices using P matrix such
        that :math:`\\Sigma_u = PP^\\prime`. P defaults to the Cholesky
        decomposition of :math:`\\Sigma_u`

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (k x k), optional
            Matrix such that Sigma_u = PP', defaults to Cholesky descomp

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        pass

    def long_run_effects(self):
        """Compute long-run effect of unit impulse

        .. math::

            \\Psi_\\infty = \\sum_{i=0}^\\infty \\Phi_i
        """
        pass

    @cache_readonly
    def _char_mat(self):
        """Characteristic matrix of the VAR"""
        pass

    def acf(self, nlags=None):
        """Compute theoretical autocovariance function

        Returns
        -------
        acf : ndarray (p x k x k)
        """
        pass

    def acorr(self, nlags=None):
        """
        Autocorrelation function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocovariance function. The
            default is the number of lags included in the model.

        Returns
        -------
        acorr : ndarray
            Autocorrelation and cross correlations (nlags, neqs, neqs)
        """
        pass

    def plot_acorr(self, nlags=10, linewidth=8):
        """Plot theoretical autocorrelation function"""
        pass

    def forecast(self, y, steps, exog_future=None):
        """Produce linear minimum MSE forecasts for desired number of steps
        ahead, using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
        steps : int

        Returns
        -------
        forecasts : ndarray (steps x neqs)

        Notes
        -----
        Lütkepohl pp 37-38
        """
        pass

    def mse(self, steps):
        """
        Compute theoretical forecast error variance matrices

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Notes
        -----
        .. math:: \\mathrm{MSE}(h) = \\sum_{i=0}^{h-1} \\Phi \\Sigma_u \\Phi^T

        Returns
        -------
        forc_covs : ndarray (steps x neqs x neqs)
        """
        pass
    forecast_cov = mse

    def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
        """
        Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------
        y : {ndarray, None}
            The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used.
        steps : int
            Number of steps ahead to forecast
        alpha : float, optional
            The significance level for the confidence intervals.
        exog_future : ndarray, optional
            Forecast values of the exogenous variables. Should include
            constant, trend, etc. as needed, including extrapolating out
            of sample.
        Returns
        -------
        point : ndarray
            Mean value of forecast
        lower : ndarray
            Lower bound of confidence interval
        upper : ndarray
            Upper bound of confidence interval

        Notes
        -----
        Lütkepohl pp. 39-40
        """
        pass

    def to_vecm(self):
        """to_vecm"""
        pass

class VARResults(VARProcess):
    """Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in
        `endog`.
    dates
    exog : ndarray

    Attributes
    ----------
    params : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    dates
    endog
    endog_lagged
    k_ar : int
        Order of VAR process
    k_trend : int
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    names : list
        variables names
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    """
    _model_type = 'VAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order, model=None, trend='c', names=None, dates=None, exog=None):
        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates
        self.n_totobs, neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(names, lag_order, k_trend, model.data.orig_exog)
        self.params = params
        self.exog = exog
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        coefs = reshaped.swapaxes(1, 2).copy()
        self.coefs_exog = params[:endog_start].T
        self.k_exog = self.coefs_exog.shape[1]
        self.k_exog_user = k_exog_user
        _params_info = {'k_trend': k_trend, 'k_exog_user': k_exog_user, 'k_ar': lag_order}
        super().__init__(coefs, self.coefs_exog, sigma_u, names=names, _params_info=_params_info)

    def plot(self):
        """Plot input time series"""
        pass

    @property
    def df_model(self):
        """
        Number of estimated parameters per variable, including the intercept / trends
        """
        pass

    @property
    def df_resid(self):
        """Number of observations minus number of estimated parameters"""
        pass

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted insample values of the response variables of the model.
        """
        pass

    @cache_readonly
    def resid(self):
        """
        Residuals of response variable resulting from estimated coefficients
        """
        pass

    def sample_acov(self, nlags=1):
        """Sample acov"""
        pass

    def sample_acorr(self, nlags=1):
        """Sample acorr"""
        pass

    def plot_sample_acorr(self, nlags=10, linewidth=8):
        """
        Plot sample autocorrelation function

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocorrelation. Does
            not count the zero lag, which will be returned.
        linewidth : int
            The linewidth for the plots.

        Returns
        -------
        Figure
            The figure that contains the plot axes.
        """
        pass

    def resid_acov(self, nlags=1):
        """
        Compute centered sample autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        pass

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        """
        pass

    @cache_readonly
    def resid_corr(self):
        """
        Centered residual correlation matrix
        """
        pass

    @cache_readonly
    def sigma_u_mle(self):
        """(Biased) maximum likelihood estimate of noise process covariance"""
        pass

    def cov_params(self):
        """Estimated variance-covariance of model coefficients

        Notes
        -----
        Covariance of vec(B), where B is the matrix
        [params_for_deterministic_terms, A_1, ..., A_p] with the shape
        (K x (Kp + number_of_deterministic_terms))
        Adjusted to be an unbiased estimator
        Ref: Lütkepohl p.74-75
        """
        pass

    def cov_ybar(self):
        """Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \\sqrt(T) (\\bar{y} - \\mu) \\rightarrow
                  {\\cal N}(0, \\Sigma_{\\bar{y}}) \\\\

            \\Sigma_{\\bar{y}} = B \\Sigma_u B^\\prime, \\text{where }
                  B = (I_K - A_1 - \\cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """
        pass

    @property
    def _cov_alpha(self):
        """
        Estimated covariance matrix of model coefficients w/o exog
        """
        pass

    @cache_readonly
    def _cov_sigma(self):
        """
        Estimated covariance matrix of vech(sigma_u)
        """
        pass

    @cache_readonly
    def llf(self):
        """Compute VAR(p) loglikelihood"""
        pass

    @cache_readonly
    def stderr(self):
        """Standard errors of coefficients, reshaped to match in size"""
        pass
    bse = stderr

    @cache_readonly
    def stderr_endog_lagged(self):
        """Stderr_endog_lagged"""
        pass

    @cache_readonly
    def stderr_dt(self):
        """Stderr_dt"""
        pass

    @cache_readonly
    def tvalues(self):
        """
        Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid) to
        test significance.
        """
        pass

    @cache_readonly
    def tvalues_endog_lagged(self):
        """tvalues_endog_lagged"""
        pass

    @cache_readonly
    def tvalues_dt(self):
        """tvalues_dt"""
        pass

    @cache_readonly
    def pvalues(self):
        """
        Two-sided p-values for model coefficients from Student t-distribution
        """
        pass

    @cache_readonly
    def pvalues_endog_lagged(self):
        """pvalues_endog_laggd"""
        pass

    @cache_readonly
    def pvalues_dt(self):
        """pvalues_dt"""
        pass

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        """
        Plot forecast
        """
        pass

    def forecast_cov(self, steps=1, method='mse'):
        """Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \\Sigma_{\\hat y}(h) = \\Sigma_y(h) + \\Omega(h) / T

        Ref: Lütkepohl pp. 96-97

        Returns
        -------
        covs : ndarray (steps x k x k)
        """
        pass

    def irf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        pass

    def irf_resim(self, orth=False, repl=1000, steps=10, seed=None, burn=100, cum=False):
        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse
           Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions
        """
        pass

    def summary(self):
        """Compute console output summary of estimates

        Returns
        -------
        summary : VARSummary
        """
        pass

    def irf(self, periods=10, var_decomp=None, var_order=None):
        """Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
        var_decomp : ndarray (k x k), lower triangular
            Must satisfy Omega = P P', where P is the passed matrix. Defaults
            to Cholesky decomposition of Omega
        var_order : sequence
            Alternate variable order for Cholesky decomposition

        Returns
        -------
        irf : IRAnalysis
        """
        pass

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Returns
        -------
        fevd : FEVD instance
        """
        pass

    def reorder(self, order):
        """Reorder variables for structural specification"""
        pass

    def test_causality(self, caused, causing=None, kind='f', signif=0.05):
        """
        Test Granger causality

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Returns
        -------
        results : CausalityTestResults

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        pass

    def test_inst_causality(self, causing, signif=0.05):
        """
        Test for instantaneous causality

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding
            variables are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level
        verbose : bool
            If True, print a table with the results.

        Returns
        -------
        results : dict
            A dict holding the test's results. The dict's keys are:

            "statistic" : float
              The calculated test statistic.

            "crit_value" : float
              The critical value of the Chi^2-distribution.

            "pvalue" : float
              The p-value corresponding to the test statistic.

            "df" : float
              The degrees of freedom of the Chi^2-distribution.

            "conclusion" : str {"reject", "fail to reject"}
              Whether H0 can be rejected or not.

            "signif" : float
              Significance level

        Notes
        -----
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".

        Instantaneous causality is a symmetric relation (i.e. if causing is
        "instantaneously causing" caused, then also caused is "instantaneously
        causing" causing), thus the naming of the parameters (which is chosen
        to be in accordance with test_granger_causality()) may be misleading.

        This method is not returning the same result as JMulTi. This is
        because the test is based on a VAR(k_ar) model in statsmodels
        (in accordance to pp. 104, 320-321 in [1]_) whereas JMulTi seems
        to be using a VAR(k_ar+1) model.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        pass

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Residual whiteness tests using Portmanteau test

        Parameters
        ----------
        nlags : int > 0
            The number of lags tested must be larger than the number of lags
            included in the VAR model.
        signif : float, between 0 and 1
            The significance level of the test.
        adjusted : bool, default False
            Flag indicating to apply small-sample adjustments.

        Returns
        -------
        WhitenessTestResults
            The test results.

        Notes
        -----
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 4.4.3.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*
           *Analysis*. Springer.
        """
        pass

    def plot_acorr(self, nlags=10, resid=True, linewidth=8):
        """
        Plot autocorrelation of sample (endog) or residuals

        Sample (Y) or Residual autocorrelations are plotted together with the
        standard :math:`2 / \\sqrt{T}` bounds.

        Parameters
        ----------
        nlags : int
            number of lags to display (excluding 0)
        resid : bool
            If True, then the autocorrelation of the residuals is plotted
            If False, then the autocorrelation of endog is plotted.
        linewidth : int
            width of vertical bars

        Returns
        -------
        Figure
            Figure instance containing the plot.
        """
        pass

    def test_normality(self, signif=0.05):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test.

        Parameters
        ----------
        signif : float
            Test significance level.

        Returns
        -------
        result : NormalityTestResults

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        pass

    @cache_readonly
    def detomega(self):
        """
        Return determinant of white noise covariance with degrees of freedom
        correction:

        .. math::

            \\hat \\Omega = \\frac{T}{T - Kp - 1} \\hat \\Omega_{\\mathrm{MLE}}
        """
        pass

    @cache_readonly
    def info_criteria(self):
        """information criteria for lagorder selection"""
        pass

    @property
    def aic(self):
        """Akaike information criterion"""
        pass

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        Lütkepohl p. 147, see info_criteria
        """
        pass

    @property
    def hqic(self):
        """Hannan-Quinn criterion"""
        pass

    @property
    def bic(self):
        """Bayesian a.k.a. Schwarz info criterion"""
        pass

    @cache_readonly
    def roots(self):
        """
        The roots of the VAR process are the solution to
        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.
        Note that the inverse roots are returned, and stability requires that
        the roots lie outside the unit circle.
        """
        pass

class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse': 'columns_eq', 'cov_params': 'cov', 'params': 'columns_eq', 'pvalues': 'columns_eq', 'tvalues': 'columns_eq', 'sigma_u': 'cov_eq', 'sigma_u_mle': 'cov_eq', 'stderr': 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'conf_int': 'multivariate_confint'}
    _wrap_methods = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(VARResultsWrapper, VARResults)

class FEVD:
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """

    def __init__(self, model, P=None, periods=None):
        self.periods = periods
        self.model = model
        self.neqs = model.neqs
        self.names = model.model.endog_names
        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)
        rng = lrange(self.neqs)
        mse = self.model.mse(periods)[:, rng, rng]
        fevd = np.empty_like(irfs)
        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T
        self.decomp = fevd.swapaxes(0, 1)

    def cov(self):
        """Compute asymptotic standard errors

        Returns
        -------
        """
        pass

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        """Plot graphical display of FEVD

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        """
        pass