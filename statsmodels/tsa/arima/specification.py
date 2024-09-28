"""
SARIMAX specification class.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import is_invertible, constrain_stationary_univariate as constrain, unconstrain_stationary_univariate as unconstrain, prepare_exog, prepare_trend_spec, prepare_trend_data
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic

class SARIMAXSpecification:
    """
    SARIMAX specification.

    Parameters
    ----------
    endog : array_like, optional
        The observed time-series process :math:`y`.
    exog : array_like, optional
        Array of exogenous regressors.
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers. May not be used in combination
        with the arguments `ar_order`, `diff`, or `ma_order`.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers. May not be used
        in combination with the arguments `seasonal_ar_order`, `seasonal_diff`,
        or `seasonal_ma_order`.
    ar_order : int or list of int
        The autoregressive order of the model. May be an integer, in which case
        all autoregressive lags up to and including it will be included.
        Alternatively, may be a list of integers specifying which lag orders
        are included. May not be used in combination with `order`.
    diff : int
        The order of integration of the model. May not be used in combination
        with `order`.
    ma_order : int or list of int
        The moving average order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` for details.
        May not be used in combination with `order`.
    seasonal_ar_order : int or list of int
        The seasonal autoregressive order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` for examples.
        Note that if `seasonal_periods = 4` and `seasonal_ar_order = 2`, then
        this implies that the overall model will include lags 4 and 8.
        May not be used in combination with `seasonal_order`.
    seasonal_diff : int
        The order of seasonal integration of the model. May not be used in
        combination with `seasonal_order`.
    seasonal_ma_order : int or list of int
        The moving average order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` and
        `seasonal_ar_order` for additional details. May not be used in
        combination with `seasonal_order`.
    seasonal_periods : int
        Number of periods in a season. May not be used in combination with
        `seasonal_order`.
    enforce_stationarity : bool, optional
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process. This is only possible in estimation by
        numerical maximum likelihood.
    enforce_invertibility : bool, optional
        Whether or not to require the moving average parameters to correspond
        to an invertible process. This is only possible in estimation by
        numerical maximum likelihood.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters by one.
        This is only applicable when considering estimation by numerical
        maximum likelihood.
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

    Attributes
    ----------
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    ar_order : int or list of int
        The autoregressive order of the model. May be an integer, in which case
        all autoregressive lags up to and including it will be included. For
        example, if `ar_order = 3`, then the model will include lags 1, 2,
        and 3. Alternatively, may be a list of integers specifying exactly
        which lag orders are included. For example, if `ar_order = [1, 3]`,
        then the model will include lags 1 and 3 but will exclude lag 2.
    diff : int
        The order of integration of the model.
    ma_order : int or list of int
        The moving average order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` for examples.
    seasonal_ar_order : int or list of int
        The seasonal autoregressive order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` for examples.
        Note that if `seasonal_periods = 4` and `seasonal_ar_order = 2`, then
        this implies that the overall model will include lags 4 and 8.
    seasonal_diff : int
        The order of seasonal integration of the model.
    seasonal_ma_order : int or list of int
        The moving average order of the model. May be an integer or
        list of integers. See the documentation for `ar_order` and
        `seasonal_ar_order` for additional details.
    seasonal_periods : int
        Number of periods in a season.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is to not
        include a trend component.
    ar_lags : list of int
        List of included autoregressive lags. If `ar_order` is a list, then
        `ar_lags == ar_order`. If `ar_lags = [1, 2]`, then the overall model
        will include the 1st and 2nd autoregressive lags.
    ma_lags : list of int
        List of included moving average lags. If `ma_order` is a list, then
        `ma_lags == ma_order`. If `ma_lags = [1, 2]`, then the overall model
        will include the 1st and 2nd moving average lags.
    seasonal_ar_lags : list of int
        List of included seasonal autoregressive lags. If `seasonal_ar_order`
        is a list, then `seasonal_ar_lags == seasonal_ar_order`. If
        `seasonal_periods = 4` and `seasonal_ar_lags = [1, 2]`, then the
        overall model will include the 4th and 8th autoregressive lags.
    seasonal_ma_lags : list of int
        List of included seasonal moving average lags. If `seasonal_ma_order`
        is a list, then `seasonal_ma_lags == seasonal_ma_order`. See the
        documentation to `seasonal_ar_lags` for examples.
    max_ar_order : int
        Largest included autoregressive lag.
    max_ma_order : int
        Largest included moving average lag.
    max_seasonal_ar_order : int
        Largest included seasonal autoregressive lag.
    max_seasonal_ma_order : int
        Largest included seasonal moving average lag.
    max_reduced_ar_order : int
        Largest lag in the reduced autoregressive polynomial. Equal to
        `max_ar_order + max_seasonal_ar_order * seasonal_periods`.
    max_reduced_ma_order : int
        Largest lag in the reduced moving average polynomial. Equal to
        `max_ma_order + max_seasonal_ma_order * seasonal_periods`.
    enforce_stationarity : bool
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. This is only possible
        in estimation by numerical maximum likelihood.
    enforce_invertibility : bool
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. This is only possible
        in estimation by numerical maximum likelihood.
    concentrate_scale : bool
        Whether or not to concentrate the variance (scale term) out of the
        log-likelihood function. This is only applicable when considering
        estimation by numerical maximum likelihood.
    is_ar_consecutive
    is_ma_consecutive
    is_integrated
    is_seasonal
    k_exog_params
    k_ar_params
    k_ma_params
    k_seasonal_ar_params
    k_seasonal_ma_params
    k_params
    exog_names
    ar_names
    ma_names
    seasonal_ar_names
    seasonal_ma_names
    param_names

    Examples
    --------
    >>> SARIMAXSpecification(order=(1, 0, 2))
    SARIMAXSpecification(endog=y, order=(1, 0, 2))

    >>> spec = SARIMAXSpecification(ar_order=1, ma_order=2)
    SARIMAXSpecification(endog=y, order=(1, 0, 2))

    >>> spec = SARIMAXSpecification(ar_order=1, seasonal_order=(1, 0, 0, 4))
    SARIMAXSpecification(endog=y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4))
    """

    def __init__(self, endog=None, exog=None, order=None, seasonal_order=None, ar_order=None, diff=None, ma_order=None, seasonal_ar_order=None, seasonal_diff=None, seasonal_ma_order=None, seasonal_periods=None, trend=None, enforce_stationarity=None, enforce_invertibility=None, concentrate_scale=None, trend_offset=1, dates=None, freq=None, missing='none', validate_specification=True):
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        has_order = order is not None
        has_specific_order = ar_order is not None or diff is not None or ma_order is not None
        has_seasonal_order = seasonal_order is not None
        has_specific_seasonal_order = seasonal_ar_order is not None or seasonal_diff is not None or seasonal_ma_order is not None or (seasonal_periods is not None)
        if has_order and has_specific_order:
            raise ValueError('Cannot specify both `order` and either of `ar_order` or `ma_order`.')
        if has_seasonal_order and has_specific_seasonal_order:
            raise ValueError('Cannot specify both `seasonal_order` and any of `seasonal_ar_order`, `seasonal_ma_order`, or `seasonal_periods`.')
        if has_specific_order:
            ar_order = 0 if ar_order is None else ar_order
            diff = 0 if diff is None else diff
            ma_order = 0 if ma_order is None else ma_order
            order = (ar_order, diff, ma_order)
        elif not has_order:
            order = (0, 0, 0)
        if has_specific_seasonal_order:
            seasonal_ar_order = 0 if seasonal_ar_order is None else seasonal_ar_order
            seasonal_diff = 0 if seasonal_diff is None else seasonal_diff
            seasonal_ma_order = 0 if seasonal_ma_order is None else seasonal_ma_order
            seasonal_periods = 0 if seasonal_periods is None else seasonal_periods
            seasonal_order = (seasonal_ar_order, seasonal_diff, seasonal_ma_order, seasonal_periods)
        elif not has_seasonal_order:
            seasonal_order = (0, 0, 0, 0)
        if len(order) != 3:
            raise ValueError('`order` argument must be an iterable with three elements.')
        if len(seasonal_order) != 4:
            raise ValueError('`seasonal_order` argument must be an iterable with four elements.')
        if validate_specification:
            if order[1] < 0:
                raise ValueError('Cannot specify negative differencing.')
            if order[1] != int(order[1]):
                raise ValueError('Cannot specify fractional differencing.')
            if seasonal_order[1] < 0:
                raise ValueError('Cannot specify negative seasonal differencing.')
            if seasonal_order[1] != int(seasonal_order[1]):
                raise ValueError('Cannot specify fractional seasonal differencing.')
            if seasonal_order[3] < 0:
                raise ValueError('Cannot specify negative seasonal periodicity.')
        order = (standardize_lag_order(order[0], 'AR'), int(order[1]), standardize_lag_order(order[2], 'MA'))
        seasonal_order = (standardize_lag_order(seasonal_order[0], 'seasonal AR'), int(seasonal_order[1]), standardize_lag_order(seasonal_order[2], 'seasonal MA'), int(seasonal_order[3]))
        if validate_specification:
            if seasonal_order[3] == 1:
                raise ValueError('Seasonal periodicity must be greater than 1.')
            if (seasonal_order[0] != 0 or seasonal_order[1] != 0 or seasonal_order[2] != 0) and seasonal_order[3] == 0:
                raise ValueError('Must include nonzero seasonal periodicity if including seasonal AR, MA, or differencing.')
        self.order = order
        self.ar_order, self.diff, self.ma_order = order
        self.seasonal_order = seasonal_order
        self.seasonal_ar_order, self.seasonal_diff, self.seasonal_ma_order, self.seasonal_periods = seasonal_order
        if isinstance(self.ar_order, list):
            self.ar_lags = self.ar_order
        else:
            self.ar_lags = np.arange(1, self.ar_order + 1).tolist()
        if isinstance(self.ma_order, list):
            self.ma_lags = self.ma_order
        else:
            self.ma_lags = np.arange(1, self.ma_order + 1).tolist()
        if isinstance(self.seasonal_ar_order, list):
            self.seasonal_ar_lags = self.seasonal_ar_order
        else:
            self.seasonal_ar_lags = np.arange(1, self.seasonal_ar_order + 1).tolist()
        if isinstance(self.seasonal_ma_order, list):
            self.seasonal_ma_lags = self.seasonal_ma_order
        else:
            self.seasonal_ma_lags = np.arange(1, self.seasonal_ma_order + 1).tolist()
        self.max_ar_order = self.ar_lags[-1] if self.ar_lags else 0
        self.max_ma_order = self.ma_lags[-1] if self.ma_lags else 0
        self.max_seasonal_ar_order = self.seasonal_ar_lags[-1] if self.seasonal_ar_lags else 0
        self.max_seasonal_ma_order = self.seasonal_ma_lags[-1] if self.seasonal_ma_lags else 0
        self.max_reduced_ar_order = self.max_ar_order + self.max_seasonal_ar_order * self.seasonal_periods
        self.max_reduced_ma_order = self.max_ma_order + self.max_seasonal_ma_order * self.seasonal_periods
        ar_lags = set(self.ar_lags)
        seasonal_ar_lags = set(np.array(self.seasonal_ar_lags) * self.seasonal_periods)
        duplicate_ar_lags = ar_lags.intersection(seasonal_ar_lags)
        if validate_specification and len(duplicate_ar_lags) > 0:
            raise ValueError('Invalid model: autoregressive lag(s) %s are in both the seasonal and non-seasonal autoregressive components.' % duplicate_ar_lags)
        ma_lags = set(self.ma_lags)
        seasonal_ma_lags = set(np.array(self.seasonal_ma_lags) * self.seasonal_periods)
        duplicate_ma_lags = ma_lags.intersection(seasonal_ma_lags)
        if validate_specification and len(duplicate_ma_lags) > 0:
            raise ValueError('Invalid model: moving average lag(s) %s are in both the seasonal and non-seasonal moving average components.' % duplicate_ma_lags)
        self.trend = trend
        self.trend_poly, _ = prepare_trend_spec(trend)
        exog_is_pandas = _is_using_pandas(exog, None)
        if validate_specification and exog is not None and (len(self.trend_poly) > 0) and (self.trend_poly[0] == 1):
            x = np.asanyarray(exog)
            ptp0 = np.ptp(x, axis=0)
            col_is_const = ptp0 == 0
            nz_const = col_is_const & (x[0] != 0)
            col_const = nz_const
            if np.any(col_const):
                raise ValueError('A constant trend was included in the model specification, but the `exog` data already contains a column of constants.')
        self.trend_terms = np.where(self.trend_poly == 1)[0]
        self.k_trend = len(self.trend_terms)
        if len(self.trend_terms) == 0:
            self.trend_order = None
            self.trend_degree = None
        elif np.all(self.trend_terms == np.arange(len(self.trend_terms))):
            self.trend_order = self.trend_terms[-1]
            self.trend_degree = self.trend_terms[-1]
        else:
            self.trend_order = self.trend_terms
            self.trend_degree = self.trend_terms[-1]
        self.k_exog, exog = prepare_exog(exog)
        faux_endog = endog is None
        if endog is None:
            endog = [] if exog is None else np.zeros(len(exog)) * np.nan
        nobs = len(endog) if exog is None else len(exog)
        if self.trend_order is not None:
            trend_data = self.construct_trend_data(nobs, trend_offset)
            if exog is None:
                exog = trend_data
            elif exog_is_pandas:
                trend_data = pd.DataFrame(trend_data, index=exog.index, columns=self.construct_trend_names())
                exog = pd.concat([trend_data, exog], axis=1)
            else:
                exog = np.c_[trend_data, exog]
        self._model = TimeSeriesModel(endog, exog=exog, dates=dates, freq=freq, missing=missing)
        self.endog = None if faux_endog else self._model.endog
        self.exog = self._model.exog
        if validate_specification and (not faux_endog) and (self.endog.ndim > 1) and (self.endog.shape[1] > 1):
            raise ValueError('SARIMAX models require univariate `endog`. Got shape %s.' % str(self.endog.shape))
        self._has_missing = None if faux_endog else np.any(np.isnan(self.endog))

    @property
    def is_ar_consecutive(self):
        """
        (bool) Is autoregressive lag polynomial consecutive.

        I.e. does it include all lags up to and including the maximum lag.
        """
        pass

    @property
    def is_ma_consecutive(self):
        """
        (bool) Is moving average lag polynomial consecutive.

        I.e. does it include all lags up to and including the maximum lag.
        """
        pass

    @property
    def is_integrated(self):
        """
        (bool) Is the model integrated.

        I.e. does it have a nonzero `diff` or `seasonal_diff`.
        """
        pass

    @property
    def is_seasonal(self):
        """(bool) Does the model include a seasonal component."""
        pass

    @property
    def k_exog_params(self):
        """(int) Number of parameters associated with exogenous variables."""
        pass

    @property
    def k_ar_params(self):
        """(int) Number of autoregressive (non-seasonal) parameters."""
        pass

    @property
    def k_ma_params(self):
        """(int) Number of moving average (non-seasonal) parameters."""
        pass

    @property
    def k_seasonal_ar_params(self):
        """(int) Number of seasonal autoregressive parameters."""
        pass

    @property
    def k_seasonal_ma_params(self):
        """(int) Number of seasonal moving average parameters."""
        pass

    @property
    def k_params(self):
        """(int) Total number of model parameters."""
        pass

    @property
    def exog_names(self):
        """(list of str) Names associated with exogenous parameters."""
        pass

    @property
    def ar_names(self):
        """(list of str) Names of (non-seasonal) autoregressive parameters."""
        pass

    @property
    def ma_names(self):
        """(list of str) Names of (non-seasonal) moving average parameters."""
        pass

    @property
    def seasonal_ar_names(self):
        """(list of str) Names of seasonal autoregressive parameters."""
        pass

    @property
    def seasonal_ma_names(self):
        """(list of str) Names of seasonal moving average parameters."""
        pass

    @property
    def param_names(self):
        """(list of str) Names of all model parameters."""
        pass

    @property
    def valid_estimators(self):
        """
        (list of str) Estimators that could be used with specification.

        Note: does not consider the presense of `exog` in determining valid
        estimators. If there are exogenous variables, then feasible Generalized
        Least Squares should be used through the `gls` estimator, and the
        `valid_estimators` are the estimators that could be passed as the
        `arma_estimator` argument to `gls`.
        """
        pass

    def validate_estimator(self, estimator):
        """
        Validate an SARIMA estimator.

        Parameters
        ----------
        estimator : str
            Name of the estimator to validate against the current state of
            the specification. Possible values are: 'yule_walker', 'burg',
            'innovations', 'hannan_rissanen', 'innovoations_mle', 'statespace'.

        Notes
        -----
        This method will raise a `ValueError` if an invalid method is passed,
        and otherwise will return None.

        This method does not consider the presense of `exog` in determining
        valid estimators. If there are exogenous variables, then feasible
        Generalized Least Squares should be used through the `gls` estimator,
        and a "valid" estimator is one that could be passed as the
        `arma_estimator` argument to `gls`.

        This method only uses the attributes `enforce_stationarity` and
        `concentrate_scale` to determine the validity of numerical maximum
        likelihood estimators. These only include 'innovations_mle' (which
        does not support `enforce_stationarity=False` or
        `concentrate_scale=True`) and 'statespace' (which supports all
        combinations of each).

        Examples
        --------
        >>> spec = SARIMAXSpecification(order=(1, 0, 2))

        >>> spec.validate_estimator('yule_walker')
        ValueError: Yule-Walker estimator does not support moving average
                    components.

        >>> spec.validate_estimator('burg')
        ValueError: Burg estimator does not support moving average components.

        >>> spec.validate_estimator('innovations')
        ValueError: Burg estimator does not support autoregressive components.

        >>> spec.validate_estimator('hannan_rissanen')  # returns None
        >>> spec.validate_estimator('innovations_mle')  # returns None
        >>> spec.validate_estimator('statespace')       # returns None

        >>> spec.validate_estimator('not_an_estimator')
        ValueError: "not_an_estimator" is not a valid estimator.
        """
        pass

    def split_params(self, params, allow_infnan=False):
        """
        Split parameter array by type into dictionary.

        Parameters
        ----------
        params : array_like
            Array of model parameters.
        allow_infnan : bool, optional
            Whether or not to allow `params` to contain -np.Inf, np.Inf, and
            np.nan. Default is False.

        Returns
        -------
        split_params : dict
            Dictionary with keys 'exog_params', 'ar_params', 'ma_params',
            'seasonal_ar_params', 'seasonal_ma_params', and (unless
            `concentrate_scale=True`) 'sigma2'. Values are the parameters
            associated with the key, based on the `params` argument.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.split_params([0.5, 4])
        {'exog_params': array([], dtype=float64),
         'ar_params': array([0.5]),
         'ma_params': array([], dtype=float64),
         'seasonal_ar_params': array([], dtype=float64),
         'seasonal_ma_params': array([], dtype=float64),
         'sigma2': 4.0}
        """
        pass

    def join_params(self, exog_params=None, ar_params=None, ma_params=None, seasonal_ar_params=None, seasonal_ma_params=None, sigma2=None):
        """
        Join parameters into a single vector.

        Parameters
        ----------
        exog_params : array_like, optional
            Parameters associated with exogenous regressors. Required if
            `exog` is part of specification.
        ar_params : array_like, optional
            Parameters associated with (non-seasonal) autoregressive component.
            Required if this component is part of the specification.
        ma_params : array_like, optional
            Parameters associated with (non-seasonal) moving average component.
            Required if this component is part of the specification.
        seasonal_ar_params : array_like, optional
            Parameters associated with seasonal autoregressive component.
            Required if this component is part of the specification.
        seasonal_ma_params : array_like, optional
            Parameters associated with seasonal moving average component.
            Required if this component is part of the specification.
        sigma2 : array_like, optional
            Innovation variance parameter. Required unless
            `concentrated_scale=True`.

        Returns
        -------
        params : ndarray
            Array of parameters.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.join_params(ar_params=0.5, sigma2=4)
        array([0.5, 4. ])
        """
        pass

    def validate_params(self, params):
        """
        Validate parameter vector by raising ValueError on invalid values.

        Parameters
        ----------
        params : array_like
            Array of model parameters.

        Notes
        -----
        Primarily checks that the parameters have the right shape and are not
        NaN or infinite. Also checks if parameters are consistent with a
        stationary process if `enforce_stationarity=True` and that they are
        consistent with an invertible process if `enforce_invertibility=True`.
        Finally, checks that the variance term is positive, unless
        `concentrate_scale=True`.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.validate_params([-0.5, 4.])  # returns None
        >>> spec.validate_params([-0.5, -2])
        ValueError: Non-positive variance term.
        >>> spec.validate_params([-1.5, 4.])
        ValueError: Non-stationary autoregressive polynomial.
        """
        pass

    def constrain_params(self, unconstrained):
        """
        Constrain parameter values to be valid through transformations.

        Parameters
        ----------
        unconstrained : array_like
            Array of model unconstrained parameters.

        Returns
        -------
        constrained : ndarray
            Array of model parameters transformed to produce a valid model.

        Notes
        -----
        This is usually only used when performing numerical minimization
        of the log-likelihood function. This function is necessary because
        the minimizers consider values over the entire real space, while
        SARIMAX models require parameters in subspaces (for example positive
        variances).

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.constrain_params([10, -2])
        array([-0.99504,  4.     ])
        """
        pass

    def unconstrain_params(self, constrained):
        """
        Reverse transformations used to constrain parameter values to be valid.

        Parameters
        ----------
        constrained : array_like
            Array of model parameters.

        Returns
        -------
        unconstrained : ndarray
            Array of parameters with constraining transformions reversed.

        Notes
        -----
        This is usually only used when performing numerical minimization
        of the log-likelihood function. This function is the (approximate)
        inverse of `constrain_params`.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.unconstrain_params([-0.5, 4.])
        array([0.57735, 2.     ])
        """
        pass

    def __repr__(self):
        """Represent SARIMAXSpecification object as a string."""
        components = []
        if self.endog is not None:
            components.append('endog=%s' % self._model.endog_names)
        if self.k_exog_params:
            components.append('exog=%s' % self.exog_names)
        components.append('order=%s' % str(self.order))
        if self.seasonal_periods > 0:
            components.append('seasonal_order=%s' % str(self.seasonal_order))
        if self.enforce_stationarity is not None:
            components.append('enforce_stationarity=%s' % self.enforce_stationarity)
        if self.enforce_invertibility is not None:
            components.append('enforce_invertibility=%s' % self.enforce_invertibility)
        if self.concentrate_scale is not None:
            components.append('concentrate_scale=%s' % self.concentrate_scale)
        return 'SARIMAXSpecification(%s)' % ', '.join(components)