"""
Linear exponential smoothing models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.validation import array_like, bool_like, float_like, string_like, int_like
from statsmodels.tsa.exponential_smoothing import initialization as es_init
from statsmodels.tsa.statespace import initialization as ss_init
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE, MEMORY_NO_FORECAST
from statsmodels.compat.pandas import Appender
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper

class ExponentialSmoothing(MLEModel):
    """
    Linear exponential smoothing models

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    trend : bool, optional
        Whether or not to include a trend component. Default is False.
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is False.
    seasonal : int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Default is
        no seasonal effects.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * 'estimated'
        * 'concentrated'
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_slope` and `initial_seasonal` if
        applicable. Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'.
    bounds : iterable[tuple], optional
        An iterable containing bounds for the parameters. Must contain four
        elements, where each element is a tuple of the form (lower, upper).
        Default is (0.0001, 0.9999) for the level, trend, and seasonal
        smoothing parameters and (0.8, 0.98) for the trend damping parameter.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood.

    Notes
    -----

    **Overview**

    The parameters and states of this model are estimated by setting up the
    exponential smoothing equations as a special case of a linear Gaussian
    state space model and applying the Kalman filter. As such, it has slightly
    worse performance than the dedicated exponential smoothing model,
    :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`, and it does not
    support multiplicative (nonlinear) exponential smoothing models.

    However, as a subclass of the state space models, this model class shares
    a consistent set of functionality with those models, which can make it
    easier to work with. In addition, it supports computing confidence
    intervals for forecasts and it supports concentrating the initial
    state out of the likelihood function.

    **Model timing**

    Typical exponential smoothing results correspond to the "filtered" output
    from state space models, because they incorporate both the transition to
    the new time point (adding the trend to the level and advancing the season)
    and updating to incorporate information from the observed datapoint. By
    contrast, the "predicted" output from state space models only incorporates
    the transition.

    One consequence is that the "initial state" corresponds to the "filtered"
    state at time t=0, but this is different from the usual state space
    initialization used in Statsmodels, which initializes the model with the
    "predicted" state at time t=1. This is important to keep in mind if
    setting the initial state directly (via `initialization_method='known'`).

    **Seasonality**

    In seasonal models, it is important to note that seasonals are included in
    the state vector of this model in the order:
    `[seasonal, seasonal.L1, seasonal.L2, seasonal.L3, ...]`. At time t, the
    `'seasonal'` state holds the seasonal factor operative at time t, while
    the `'seasonal.L'` state holds the seasonal factor that would have been
    operative at time t-1.

    Suppose that the seasonal order is `n_seasons = 4`. Then, because the
    initial state corresponds to time t=0 and the time t=1 is in the same
    season as time t=-3, the initial seasonal factor for time t=1 comes from
    the lag "L3" initial seasonal factor (i.e. at time t=1 this will be both
    the "L4" seasonal factor as well as the "L0", or current, seasonal factor).

    When the initial state is estimated (`initialization_method='estimated'`),
    there are only `n_seasons - 1` parameters, because the seasonal factors are
    normalized to sum to one. The three parameters that are estimated
    correspond to the lags "L0", "L1", and "L2" seasonal factors as of time
    t=0 (alternatively, the lags "L1", "L2", and "L3" as of time t=1).

    When the initial state is given (`initialization_method='known'`), the
    initial seasonal factors for time t=0 must be given by the argument
    `initial_seasonal`. This can either be a length `n_seasons - 1` array --
    in which case it should contain the lags "L0" - "L2" (in that order)
    seasonal factors as of time t=0 -- or a length `n_seasons` array, in which
    case it should contain the "L0" - "L3" (in that order) seasonal factors
    as of time t=0.

    Note that in the state vector and parameters, the "L0" seasonal is
    called "seasonal" or "initial_seasonal", while the i>0 lag is
    called "seasonal.L{i}".

    References
    ----------
    [1] Hyndman, Rob, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder.
        Forecasting with exponential smoothing: the state space approach.
        Springer Science & Business Media, 2008.
    """

    def __init__(self, endog, trend=False, damped_trend=False, seasonal=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, bounds=None, concentrate_scale=True, dates=None, freq=None):
        self.trend = bool_like(trend, 'trend')
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        self.seasonal_periods = int_like(seasonal, 'seasonal', optional=True)
        self.seasonal = self.seasonal_periods is not None
        self.initialization_method = string_like(initialization_method, 'initialization_method').lower()
        self.concentrate_scale = bool_like(concentrate_scale, 'concentrate_scale')
        self.bounds = bounds
        if self.bounds is None:
            self.bounds = [(0.0001, 1 - 0.0001)] * 3 + [(0.8, 0.98)]
        if self.seasonal_periods == 1:
            raise ValueError('Cannot have a seasonal period of 1.')
        if self.seasonal and self.seasonal_periods is None:
            raise NotImplementedError('Unable to detect season automatically; please specify `seasonal_periods`.')
        if self.initialization_method not in ['concentrated', 'estimated', 'simple', 'heuristic', 'known']:
            raise ValueError('Invalid initialization method "%s".' % initialization_method)
        if self.initialization_method == 'known':
            if initial_level is None:
                raise ValueError('`initial_level` argument must be provided when initialization method is set to "known".')
            if initial_trend is None and self.trend:
                raise ValueError('`initial_trend` argument must be provided for models with a trend component when initialization method is set to "known".')
            if initial_seasonal is None and self.seasonal:
                raise ValueError('`initial_seasonal` argument must be provided for models with a seasonal component when initialization method is set to "known".')
        if not self.seasonal or self.seasonal_periods is None:
            self._seasonal_periods = 0
        else:
            self._seasonal_periods = self.seasonal_periods
        k_states = 2 + int(self.trend) + self._seasonal_periods
        k_posdef = 1
        init = ss_init.Initialization(k_states, 'known', constant=[0] * k_states)
        super(ExponentialSmoothing, self).__init__(endog, k_states=k_states, k_posdef=k_posdef, initialization=init, dates=dates, freq=freq)
        if self.concentrate_scale:
            self.ssm.filter_concentrated = True
        self.ssm['design', 0, 0] = 1.0
        self.ssm['selection', 0, 0] = 1.0
        self.ssm['state_cov', 0, 0] = 1.0
        self.ssm['design', 0, 1] = 1.0
        self.ssm['transition', 1, 1] = 1.0
        if self.trend:
            self.ssm['transition', 1:3, 2] = 1.0
        if self.seasonal:
            k = 2 + int(self.trend)
            self.ssm['design', 0, k] = 1.0
            self.ssm['transition', k, -1] = 1.0
            self.ssm['transition', k + 1:k_states, k:k_states - 1] = np.eye(self.seasonal_periods - 1)
        if self.initialization_method != 'known':
            msg = 'Cannot give `%%s` argument when initialization is "%s"' % initialization_method
            if initial_level is not None:
                raise ValueError(msg % 'initial_level')
            if initial_trend is not None:
                raise ValueError(msg % 'initial_trend')
            if initial_seasonal is not None:
                raise ValueError(msg % 'initial_seasonal')
        if self.initialization_method == 'simple':
            initial_level, initial_trend, initial_seasonal = es_init._initialization_simple(self.endog[:, 0], trend='add' if self.trend else None, seasonal='add' if self.seasonal else None, seasonal_periods=self.seasonal_periods)
        elif self.initialization_method == 'heuristic':
            initial_level, initial_trend, initial_seasonal = es_init._initialization_heuristic(self.endog[:, 0], trend='add' if self.trend else None, seasonal='add' if self.seasonal else None, seasonal_periods=self.seasonal_periods)
        elif self.initialization_method == 'known':
            initial_level = float_like(initial_level, 'initial_level')
            if self.trend:
                initial_trend = float_like(initial_trend, 'initial_trend')
            if self.seasonal:
                initial_seasonal = array_like(initial_seasonal, 'initial_seasonal')
                if len(initial_seasonal) == self.seasonal_periods - 1:
                    initial_seasonal = np.r_[initial_seasonal, 0 - np.sum(initial_seasonal)]
                if len(initial_seasonal) != self.seasonal_periods:
                    raise ValueError('Invalid length of initial seasonal values. Must be one of s or s-1, where s is the number of seasonal periods.')
        methods = ['simple', 'heuristic']
        if self.initialization_method in methods and initial_seasonal is not None:
            initial_seasonal = initial_seasonal[::-1]
        self._initial_level = initial_level
        self._initial_trend = initial_trend
        self._initial_seasonal = initial_seasonal
        self._initial_state = None
        methods = ['simple', 'heuristic', 'known']
        if not self.damped_trend and self.initialization_method in methods:
            self._initialize_constant_statespace(initial_level, initial_trend, initial_seasonal)
        self._init_keys += ['trend', 'damped_trend', 'seasonal', 'initialization_method', 'initial_level', 'initial_trend', 'initial_seasonal', 'bounds', 'concentrate_scale', 'dates', 'freq']

class ExponentialSmoothingResults(MLEResults):
    """
    Results from fitting a linear exponential smoothing model
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        self.initial_state = model._initial_state
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            self.initial_state = pd.DataFrame([model._initial_state], columns=model.state_names[1:])
            if model._index_dates and model._index_freq is not None:
                self.initial_state.index = index.shift(-1)[:1]

class ExponentialSmoothingResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ExponentialSmoothingResultsWrapper, ExponentialSmoothingResults)