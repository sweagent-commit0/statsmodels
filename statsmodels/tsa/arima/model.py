"""
ARIMA model class.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.tools import diff
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import innovations, innovations_mle
from statsmodels.tsa.arima.estimators.gls import gls as estimate_gls
from statsmodels.tsa.arima.specification import SARIMAXSpecification

class ARIMA(sarimax.SARIMAX):
    """
    Autoregressive Integrated Moving Average (ARIMA) model, and extensions

    This model is the basic interface for ARIMA-type models, including those
    with exogenous regressors and those with seasonal components. The most
    general form of the model is SARIMAX(p, d, q)x(P, D, Q, s). It also allows
    all specialized cases, including

    - autoregressive models: AR(p)
    - moving average models: MA(q)
    - mixed autoregressive moving average models: ARMA(p, q)
    - integration models: ARIMA(p, d, q)
    - seasonal models: SARIMA(P, D, Q, s)
    - regression with errors that follow one of the above ARIMA-type models

    Parameters
    ----------
    endog : array_like, optional
        The observed time-series process :math:`y`.
    exog : array_like, optional
        Array of exogenous regressors.
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend. Can be specified as a
        string where 'c' indicates a constant term, 't' indicates a
        linear trend in time, and 'ct' includes both. Can also be specified as
        an iterable defining a polynomial, as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is 'c' for
        models without integration, and no trend for models with integration.
        Note that all trend terms are included in the model as exogenous
        regressors, which differs from how trends are included in ``SARIMAX``
        models.  See the Notes section for a precise definition of the
        treatment of trend terms.
    enforce_stationarity : bool, optional
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process.
    enforce_invertibility : bool, optional
        Whether or not to require the moving average parameters to correspond
        to an invertible process.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters by one.
        This is only applicable when considering estimation by numerical
        maximum likelihood.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    dates : array_like of datetime, optional
        If no index is given by `endog` or `exog`, an array-like object of
        datetime objects can be provided.
    freq : str, optional
        If no index is given by `endog` or `exog`, the frequency of the
        time-series may be specified here as a Pandas offset or offset string.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    This model incorporates both exogenous regressors and trend components
    through "regression with ARIMA errors". This differs from the
    specification estimated using ``SARIMAX`` which treats the trend
    components separately from any included exogenous regressors. The full
    specification of the model estimated here is:

    .. math::

        Y_{t}-\\delta_{0}-\\delta_{1}t-\\ldots-\\delta_{k}t^{k}-X_{t}\\beta
            & =\\epsilon_{t} \\\\
        \\left(1-L\\right)^{d}\\left(1-L^{s}\\right)^{D}\\Phi\\left(L\\right)
        \\Phi_{s}\\left(L\\right)\\epsilon_{t}
            & =\\Theta\\left(L\\right)\\Theta_{s}\\left(L\\right)\\eta_{t}

    where :math:`\\eta_t \\sim WN(0,\\sigma^2)` is a white noise process, L
    is the lag operator, and :math:`G(L)` are lag polynomials corresponding
    to the autoregressive (:math:`\\Phi`), seasonal autoregressive
    (:math:`\\Phi_s`), moving average (:math:`\\Theta`), and seasonal moving
    average components (:math:`\\Theta_s`).

    `enforce_stationarity` and `enforce_invertibility` are specified in the
    constructor because they affect loglikelihood computations, and so should
    not be changed on the fly. This is why they are not instead included as
    arguments to the `fit` method.

    See the notebook `ARMA: Sunspots Data
    <../examples/notebooks/generated/tsa_arma_0.html>`__ and
    `ARMA: Artificial Data <../examples/notebooks/generated/tsa_arma_1.html>`__
    for an overview.

    .. todo:: should concentrate_scale=True by default

    Examples
    --------
    >>> mod = sm.tsa.arima.ARIMA(endog, order=(1, 0, 0))
    >>> res = mod.fit()
    >>> print(res.summary())
    """

    def __init__(self, endog, exog=None, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=False, trend_offset=1, dates=None, freq=None, missing='none', validate_specification=True):
        integrated = order[1] > 0 or seasonal_order[1] > 0
        if trend is None and (not integrated):
            trend = 'c'
        elif trend is None:
            trend = 'n'
        self._spec_arima = SARIMAXSpecification(endog, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=None, enforce_invertibility=None, concentrate_scale=concentrate_scale, trend_offset=trend_offset, dates=dates, freq=freq, missing=missing, validate_specification=validate_specification)
        exog = self._spec_arima._model.data.orig_exog
        has_trend = len(self._spec_arima.trend_terms) > 0
        if has_trend:
            lowest_trend = np.min(self._spec_arima.trend_terms)
            if lowest_trend < order[1] + seasonal_order[1]:
                raise ValueError('In models with integration (`d > 0`) or seasonal integration (`D > 0`), trend terms of lower order than `d + D` cannot be (as they would be eliminated due to the differencing operation). For example, a constant cannot be included in an ARIMA(1, 1, 1) model, but including a linear trend, which would have the same effect as fitting a constant to the differenced data, is allowed.')
        input_exog = None
        if exog is not None:
            if _is_using_pandas(exog, None):
                input_exog = exog.iloc[:, self._spec_arima.k_trend:]
            else:
                input_exog = exog[:, self._spec_arima.k_trend:]
        super().__init__(endog, exog, trend=None, order=order, seasonal_order=seasonal_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility, concentrate_scale=concentrate_scale, dates=dates, freq=freq, missing=missing, validate_specification=validate_specification)
        self.trend = trend
        self._input_exog = input_exog
        if exog is not None:
            self._input_exog_names = self.exog_names[self._spec_arima.k_trend:]
        else:
            self._input_exog_names = None
        self.k_exog = self._spec_arima.k_exog
        self.k_trend = self._spec_arima.k_trend
        unused = ['measurement_error', 'time_varying_regression', 'mle_regression', 'simple_differencing', 'hamilton_representation']
        self._init_keys = [key for key in self._init_keys if key not in unused]

    def fit(self, start_params=None, transformed=True, includes_fixed=False, method=None, method_kwargs=None, gls=None, gls_kwargs=None, cov_type=None, cov_kwds=None, return_params=False, low_memory=False):
        """
        Fit (estimate) the parameters of the model.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        method : str, optional
            The method used for estimating the parameters of the model. Valid
            options include 'statespace', 'innovations_mle', 'hannan_rissanen',
            'burg', 'innovations', and 'yule_walker'. Not all options are
            available for every specification (for example 'yule_walker' can
            only be used with AR(p) models).
        method_kwargs : dict, optional
            Arguments to pass to the fit function for the parameter estimator
            described by the `method` argument.
        gls : bool, optional
            Whether or not to use generalized least squares (GLS) to estimate
            regression effects. The default is False if `method='statespace'`
            and is True otherwise.
        gls_kwargs : dict, optional
            Arguments to pass to the GLS estimation fit method. Only applicable
            if GLS estimation is used (see `gls` argument for details).
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'opg' unless memory conservation is used to avoid
            computing the loglikelihood values for each observation, in which
            case the default is 'oim'.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including smoothed results and in-sample
            prediction), although out-of-sample forecasting is possible.
            Default is False.

        Returns
        -------
        ARIMAResults

        Examples
        --------
        >>> mod = sm.tsa.arima.ARIMA(endog, order=(1, 0, 0))
        >>> res = mod.fit()
        >>> print(res.summary())
        """
        pass

@Appender(sarimax.SARIMAXResults.__doc__)
class ARIMAResults(sarimax.SARIMAXResults):
    pass

class ARIMAResultsWrapper(sarimax.SARIMAXResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(sarimax.SARIMAXResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(sarimax.SARIMAXResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ARIMAResultsWrapper, ARIMAResults)