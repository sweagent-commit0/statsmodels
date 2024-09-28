"""
Notes
-----
Code written using below textbook as a reference.
Results are checked against the expected outcomes in the text book.

Properties:
Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and
practice. OTexts, 2014.

Author: Terence L van Zyl
Modified: Kevin Sheppard
"""
from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any, Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import array_like, bool_like, dict_like, float_like, int_like, string_like
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import _initialization_heuristic, _initialization_simple
from statsmodels.tsa.holtwinters import _exponential_smoothers as smoothers, _smoothers as py_smoothers
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import to_restricted, to_unrestricted
from statsmodels.tsa.holtwinters.results import HoltWintersResults, HoltWintersResultsWrapper
from statsmodels.tsa.tsatools import freq_to_period
SMOOTHERS = {('mul', 'add'): smoothers.holt_win_add_mul_dam, ('mul', 'mul'): smoothers.holt_win_mul_mul_dam, ('mul', None): smoothers.holt_win__mul, ('add', 'add'): smoothers.holt_win_add_add_dam, ('add', 'mul'): smoothers.holt_win_mul_add_dam, ('add', None): smoothers.holt_win__add, (None, 'add'): smoothers.holt_add_dam, (None, 'mul'): smoothers.holt_mul_dam, (None, None): smoothers.holt__}
PY_SMOOTHERS = {('mul', 'add'): py_smoothers.holt_win_add_mul_dam, ('mul', 'mul'): py_smoothers.holt_win_mul_mul_dam, ('mul', None): py_smoothers.holt_win__mul, ('add', 'add'): py_smoothers.holt_win_add_add_dam, ('add', 'mul'): py_smoothers.holt_win_mul_add_dam, ('add', None): py_smoothers.holt_win__add, (None, 'add'): py_smoothers.holt_add_dam, (None, 'mul'): py_smoothers.holt_mul_dam, (None, None): py_smoothers.holt__}

class _OptConfig:
    alpha: float
    beta: float
    phi: float
    gamma: float
    level: float
    trend: float
    seasonal: np.ndarray
    y: np.ndarray
    params: np.ndarray
    mask: np.ndarray
    mle_retvals: Any

class ExponentialSmoothing(TimeSeriesModel):
    """
    Holt Winter's Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use the value as lambda.
    bounds : dict[str, tuple[float, float]], optional
        An dictionary containing bounds for the parameters in the model,
        excluding the initial values if estimated. The keys of the dictionary
        are the variable names, e.g., smoothing_level or initial_slope.
        The initial seasonal variables are labeled initial_seasonal.<j>
        for j=0,...,m-1 where m is the number of period in a full season.
        Use None to indicate a non-binding constraint, e.g., (0, None)
        constrains a parameter to be non-negative.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    This is a full implementation of the holt winters exponential smoothing as
    per [1]_. This includes all the unstable methods as well as the stable
    methods. The implementation of the library covers the functionality of the
    R library as much as possible whilst still being Pythonic.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    @deprecate_kwarg('damped', 'damped_trend')
    def __init__(self, endog, trend=None, damped_trend=False, seasonal=None, *, seasonal_periods=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, use_boxcox=False, bounds=None, dates=None, freq=None, missing='none'):
        super().__init__(endog, None, dates, freq, missing=missing)
        self._y = self._data = array_like(endog, 'endog', ndim=1, contiguous=True, order='C')
        options = ('add', 'mul', 'additive', 'multiplicative')
        trend = string_like(trend, 'trend', options=options, optional=True)
        if trend in ['additive', 'multiplicative']:
            trend = {'additive': 'add', 'multiplicative': 'mul'}[trend]
        self.trend = trend
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        seasonal = string_like(seasonal, 'seasonal', options=options, optional=True)
        if seasonal in ['additive', 'multiplicative']:
            seasonal = {'additive': 'add', 'multiplicative': 'mul'}[seasonal]
        self.seasonal = seasonal
        self.has_trend = trend in ['mul', 'add']
        self.has_seasonal = seasonal in ['mul', 'add']
        if (self.trend == 'mul' or self.seasonal == 'mul') and (not np.all(self._data > 0.0)):
            raise ValueError('endog must be strictly positive when usingmultiplicative trend or seasonal components.')
        if self.damped_trend and (not self.has_trend):
            raise ValueError('Can only dampen the trend component')
        if self.has_seasonal:
            self.seasonal_periods = int_like(seasonal_periods, 'seasonal_periods', optional=True)
            if seasonal_periods is None:
                try:
                    self.seasonal_periods = freq_to_period(self._index_freq)
                except Exception:
                    raise ValueError('seasonal_periods has not been provided and index does not have a known freq. You must provide seasonal_periods')
            if self.seasonal_periods <= 1:
                raise ValueError('seasonal_periods must be larger than 1.')
            assert self.seasonal_periods is not None
        else:
            self.seasonal_periods = 0
        self.nobs = len(self.endog)
        options = ('known', 'estimated', 'heuristic', 'legacy-heuristic')
        self._initialization_method = string_like(initialization_method, 'initialization_method', optional=False, options=options)
        self._initial_level = float_like(initial_level, 'initial_level', optional=True)
        self._initial_trend = float_like(initial_trend, 'initial_trend', optional=True)
        self._initial_seasonal = array_like(initial_seasonal, 'initial_seasonal', optional=True)
        estimated = self._initialization_method == 'estimated'
        self._estimate_level = estimated
        self._estimate_trend = estimated and self.trend is not None
        self._estimate_seasonal = estimated and self.seasonal is not None
        self._bounds = self._check_bounds(bounds)
        self._use_boxcox = use_boxcox
        self._lambda = np.nan
        self._y = self._boxcox()
        self._initialize()
        self._fixed_parameters = {}

    @contextlib.contextmanager
    def fix_params(self, values):
        """
        Temporarily fix parameters for estimation.

        Parameters
        ----------
        values : dict
            Values to fix. The key is the parameter name and the value is the
            fixed value.

        Yields
        ------
        None
            No value returned.

        Examples
        --------
        >>> from statsmodels.datasets.macrodata import load_pandas
        >>> data = load_pandas()
        >>> import statsmodels.tsa.api as tsa
        >>> mod = tsa.ExponentialSmoothing(data.data.realcons, trend="add",
        ...                                initialization_method="estimated")
        >>> with mod.fix_params({"smoothing_level": 0.2}):
        ...     mod.fit()
        """
        pass

    def predict(self, params, start=None, end=None):
        """
        In-sample and out-of-sample prediction.

        Parameters
        ----------
        params : ndarray
            The fitted model parameters.
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.

        Returns
        -------
        ndarray
            The predicted values.
        """
        pass

    @deprecate_kwarg('smoothing_slope', 'smoothing_trend')
    @deprecate_kwarg('initial_slope', 'initial_trend')
    @deprecate_kwarg('damping_slope', 'damping_trend')
    def fit(self, smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, damping_trend=None, *, optimized=True, remove_bias=False, start_params=None, method=None, minimize_kwargs=None, use_brute=True, use_boxcox=None, use_basinhopping=None, initial_level=None, initial_trend=None):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal : float, optional
            The gamma value of the holt winters seasonal method, if the value
            is set then this value will be used as the value.
        damping_trend : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        start_params : array_like, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data. See
            the notes for the structure of the model parameters.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" , "TNC",
            "SLSQP" (default), "Powell", "trust-constr", "basinhopping" (also
            "bh") and "least_squares" (also "ls"). basinhopping tries multiple
            starting values in an attempt to find a global minimizer in
            non-convex problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B", "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares functions. The valid keywords are optimizer
            specific. Consult SciPy's documentation for the full set of
            options.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.

            .. deprecated:: 0.12

               Set use_boxcox when constructing the model

        use_basinhopping : bool, optional
            Deprecated. Using Basin Hopping optimizer to find optimal values.
            Use ``method`` instead.

            .. deprecated:: 0.12

               Use ``method`` instead.

        initial_level : float, optional
            Value to use when initializing the fitted level.

            .. deprecated:: 0.12

               Set initial_level when constructing the model

        initial_trend : float, optional
            Value to use when initializing the fitted trend.

            .. deprecated:: 0.12

               Set initial_trend when constructing the model
               or set initialization_method.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing
        as per [1]. This includes all the unstable methods as well as the
        stable methods. The implementation of the library covers the
        functionality of the R library as much as possible whilst still
        being Pythonic.

        The parameters are ordered

        [alpha, beta, gamma, initial_level, initial_trend, phi]

        which are then followed by m seasonal values if the model contains
        a seasonal smoother. Any parameter not relevant for the model is
        omitted. For example, a model that has a level and a seasonal
        component, but no trend and is not damped, would have starting
        values

        [alpha, gamma, initial_level, s0, s1, ..., s<m-1>]

        where sj is the initial value for seasonal component j.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        pass

    def initial_values(self, initial_level=None, initial_trend=None, force=False):
        """
        Compute initial values used in the exponential smoothing recursions.

        Parameters
        ----------
        initial_level : {float, None}
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        force : bool
            Force the calculation even if initial values exist.

        Returns
        -------
        initial_level : float
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        initial_seasons : list
            The initial values used for the seasonal components.

        Notes
        -----
        Convenience function the exposes the values used to initialize the
        recursions. When optimizing parameters these are used as starting
        values.

        Method used to compute the initial value depends on when components
        are included in the model.  In a simple exponential smoothing model
        without trend or a seasonal components, the initial value is set to the
        first observation. When a trend is added, the trend is initialized
        either using y[1]/y[0], if multiplicative, or y[1]-y[0]. When the
        seasonal component is added the initialization adapts to account for
        the modified structure.
        """
        pass

    @deprecate_kwarg('smoothing_slope', 'smoothing_trend')
    @deprecate_kwarg('damping_slope', 'damping_trend')
    def _predict(self, h=None, smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, initial_level=None, initial_trend=None, damping_trend=None, initial_seasons=None, use_boxcox=None, lamda=None, remove_bias=None, is_optimized=None):
        """
        Helper prediction function

        Parameters
        ----------
        h : int, optional
            The number of time steps to forecast ahead.
        """
        pass

class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.

    See Also
    --------
    ExponentialSmoothing
        Exponential smoothing with trend and seasonal components.
    Holt
        Exponential smoothing with a trend component.

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as
    per [1]_.  `SimpleExpSmoothing` is a restricted version of
    :class:`ExponentialSmoothing`.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, endog, initialization_method=None, initial_level=None):
        super().__init__(endog, initialization_method=initialization_method, initial_level=initial_level)

    def fit(self, smoothing_level=None, *, optimized=True, start_params=None, initial_level=None, use_brute=True, use_boxcox=None, remove_bias=False, method=None, minimize_kwargs=None):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The smoothing_level value of the simple exponential smoothing, if
            the value is set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data.
        initial_level : float, optional
            Value to use when initializing the fitted level.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
            "least_squares" (also "ls"). basinhopping tries multiple starting
            values in an attempt to find a global minimizer in non-convex
            problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares. The valid keywords are optimizer specific.
            Consult SciPy's documentation for the full set of options.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the simple exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        pass

class Holt(ExponentialSmoothing):
    """
    Holt's Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    exponential : bool, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.

    See Also
    --------
    ExponentialSmoothing
        Exponential smoothing with trend and seasonal components.
    SimpleExpSmoothing
        Basic exponential smoothing with only a level component.

    Notes
    -----
    This is a full implementation of the Holt's exponential smoothing as
    per [1]_. `Holt` is a restricted version of :class:`ExponentialSmoothing`.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    @deprecate_kwarg('damped', 'damped_trend')
    def __init__(self, endog, exponential=False, damped_trend=False, initialization_method=None, initial_level=None, initial_trend=None):
        trend = 'mul' if exponential else 'add'
        super().__init__(endog, trend=trend, damped_trend=damped_trend, initialization_method=initialization_method, initial_level=initial_level, initial_trend=initial_trend)

    @deprecate_kwarg('smoothing_slope', 'smoothing_trend')
    @deprecate_kwarg('initial_slope', 'initial_trend')
    @deprecate_kwarg('damping_slope', 'damping_trend')
    def fit(self, smoothing_level=None, smoothing_trend=None, *, damping_trend=None, optimized=True, start_params=None, initial_level=None, initial_trend=None, use_brute=True, use_boxcox=None, remove_bias=False, method=None, minimize_kwargs=None):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        damping_trend : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data.
        initial_level : float, optional
            Value to use when initializing the fitted level.

            .. deprecated:: 0.12

               Set initial_level when constructing the model

        initial_trend : float, optional
            Value to use when initializing the fitted trend.

            .. deprecated:: 0.12

               Set initial_trend when constructing the model

        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
            "least_squares" (also "ls"). basinhopping tries multiple starting
            values in an attempt to find a global minimizer in non-convex
            problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares. The valid keywords are optimizer specific.
            Consult SciPy's documentation for the full set of options.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the Holt's exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        pass