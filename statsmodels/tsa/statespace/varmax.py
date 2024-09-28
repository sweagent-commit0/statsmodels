"""
Vector Autoregressive Moving Average with eXogenous regressors model

Author: Chad Fulton
License: Simplified-BSD
"""
import contextlib
from warnings import warn
import pandas as pd
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import is_invertible, concat, prepare_exog, constrain_stationary_multivariate, unconstrain_stationary_multivariate, prepare_trend_spec, prepare_trend_data

class VARMAX(MLEModel):
    """
    Vector Autoregressive Moving Average with eXogenous regressors model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`, , shaped nobs x k_endog.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is a constant trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):

    .. math::

        y_t = A(t) + A_1 y_{t-1} + \\dots + A_p y_{t-p} + B x_t + \\epsilon_t +
        M_1 \\epsilon_{t-1} + \\dots M_q \\epsilon_{t-q}

    where :math:`\\epsilon_t \\sim N(0, \\Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
    where the variables are measured with error.

    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\\{A_i, M_j\\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more information. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, exog=None, order=(1, 0), trend='c', error_cov_type='unstructured', measurement_error=False, enforce_stationarity=True, enforce_invertibility=True, trend_offset=1, **kwargs):
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.order = order
        self.k_ar = int(order[0])
        self.k_ma = int(order[1])
        if error_cov_type not in ['diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type specification.')
        if self.k_ar == 0 and self.k_ma == 0:
            raise ValueError('Invalid VARMAX(p,q) specification; at least one p,q must be greater than zero.')
        if self.k_ar > 0 and self.k_ma > 0:
            warn('Estimation of VARMA(p,q) models is not generically robust, due especially to identification issues.', EstimationWarning)
        self.trend = trend
        self.trend_offset = trend_offset
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._trend_is_const = self.polynomial_trend.size == 1 and self.polynomial_trend[0] == 1
        self.k_exog, exog = prepare_exog(exog)
        self.mle_regression = self.k_exog > 0
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)
        _min_k_ar = max(self.k_ar, 1)
        self._k_order = _min_k_ar + self.k_ma
        k_endog = endog.shape[1]
        k_posdef = k_endog
        k_states = k_endog * self._k_order
        kwargs.setdefault('initialization', 'stationary')
        kwargs.setdefault('inversion_method', INVERT_UNIVARIATE | SOLVE_LU)
        super(VARMAX, self).__init__(endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        if self.k_exog > 0 or (self.k_trend > 0 and (not self._trend_is_const)):
            self.ssm._time_invariant = False
        self.parameters = {}
        self.parameters['trend'] = self.k_endog * self.k_trend
        self.parameters['ar'] = self.k_endog ** 2 * self.k_ar
        self.parameters['ma'] = self.k_endog ** 2 * self.k_ma
        self.parameters['regression'] = self.k_endog * self.k_exog
        if self.error_cov_type == 'diagonal':
            self.parameters['state_cov'] = self.k_endog
        elif self.error_cov_type == 'unstructured':
            self.parameters['state_cov'] = int(self.k_endog * (self.k_endog + 1) / 2)
        self.parameters['obs_cov'] = self.k_endog * self.measurement_error
        self.k_params = sum(self.parameters.values())
        trend_data = prepare_trend_data(self.polynomial_trend, self.k_trend, self.nobs + 1, offset=self.trend_offset)
        self._trend_data = trend_data[:-1]
        self._final_trend = trend_data[-1:]
        if self.k_trend > 0 and (not self._trend_is_const) or self.k_exog > 0:
            self.ssm['state_intercept'] = np.zeros((self.k_states, self.nobs))
        idx = np.diag_indices(self.k_endog)
        self.ssm[('design',) + idx] = 1
        if self.k_ar > 0:
            idx = np.diag_indices((self.k_ar - 1) * self.k_endog)
            idx = (idx[0] + self.k_endog, idx[1])
            self.ssm[('transition',) + idx] = 1
        idx = np.diag_indices((self.k_ma - 1) * self.k_endog)
        idx = (idx[0] + (_min_k_ar + 1) * self.k_endog, idx[1] + _min_k_ar * self.k_endog)
        self.ssm[('transition',) + idx] = 1
        idx = np.diag_indices(self.k_endog)
        self.ssm[('selection',) + idx] = 1
        idx = (idx[0] + _min_k_ar * self.k_endog, idx[1])
        if self.k_ma > 0:
            self.ssm[('selection',) + idx] = 1
        if self._trend_is_const and self.k_exog == 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :]
        elif self.k_trend > 0 or self.k_exog > 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :-1]
        if self.k_ar > 0:
            self._idx_transition = np.s_['transition', :k_endog, :]
        else:
            self._idx_transition = np.s_['transition', :k_endog, k_endog:]
        if self.error_cov_type == 'diagonal':
            self._idx_state_cov = ('state_cov',) + np.diag_indices(self.k_endog)
        elif self.error_cov_type == 'unstructured':
            self._idx_lower_state_cov = np.tril_indices(self.k_endog)
        if self.measurement_error:
            self._idx_obs_cov = ('obs_cov',) + np.diag_indices(self.k_endog)

        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return (param_slice, offset)
        offset = 0
        self._params_trend, offset = _slice('trend', offset)
        self._params_ar, offset = _slice('ar', offset)
        self._params_ma, offset = _slice('ma', offset)
        self._params_regression, offset = _slice('regression', offset)
        self._params_state_cov, offset = _slice('state_cov', offset)
        self._params_obs_cov, offset = _slice('obs_cov', offset)
        self._final_exog = None
        self._init_keys += ['order', 'trend', 'error_cov_type', 'measurement_error', 'enforce_stationarity', 'enforce_invertibility'] + list(kwargs.keys())

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        Constrains the factor transition to be stationary and variances to be
        positive.
        """
        pass

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        pass

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for simulating or forecasting with `exog` or
        trend, because if we had these then the last predicted_state has been
        set to NaN since we did not have the appropriate `exog` to create it.
        Since we handle trend in the same way as `exog`, we still have this
        issue when only trend is used without `exog`.
        """
        pass

class VARMAXResults(MLEResults):
    """
    Class to hold results from fitting an VARMAX model.

    Parameters
    ----------
    model : VARMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.
    coefficient_matrices_vma : ndarray
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, cov_kwds=None, **kwargs):
        super(VARMAXResults, self).__init__(model, params, filter_results, cov_type, cov_kwds, **kwargs)
        self.specification = Bunch(**{'error_cov_type': self.model.error_cov_type, 'measurement_error': self.model.measurement_error, 'enforce_stationarity': self.model.enforce_stationarity, 'enforce_invertibility': self.model.enforce_invertibility, 'trend_offset': self.model.trend_offset, 'order': self.model.order, 'k_ar': self.model.k_ar, 'k_ma': self.model.k_ma, 'trend': self.model.trend, 'k_trend': self.model.k_trend, 'k_exog': self.model.k_exog})
        self.coefficient_matrices_var = None
        self.coefficient_matrices_vma = None
        if self.model.k_ar > 0:
            ar_params = np.array(self.params[self.model._params_ar])
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            self.coefficient_matrices_var = ar_params.reshape(k_endog * k_ar, k_endog).T.reshape(k_endog, k_endog, k_ar).T
        if self.model.k_ma > 0:
            ma_params = np.array(self.params[self.model._params_ma])
            k_endog = self.model.k_endog
            k_ma = self.model.k_ma
            self.coefficient_matrices_vma = ma_params.reshape(k_endog * k_ma, k_endog).T.reshape(k_endog, k_endog, k_ma).T

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        This context manager calls the model-level context manager and
        additionally updates the last element of filter_results.state_intercept
        appropriately.
        """
        pass

    @contextlib.contextmanager
    def _set_final_predicted_state(self, exog, out_of_sample):
        """
        Set the final predicted state value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for forecasting with `exog`, because
        if we had these then the last predicted_state has been set to NaN since
        we did not have the appropriate `exog` to create it.
        """
        pass

class VARMAXResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(VARMAXResultsWrapper, VARMAXResults)