"""
ETS models for time series analysis.

The ETS models are a family of time series models. They can be seen as a
generalization of simple exponential smoothing to time series that contain
trends and seasonalities. Additionally, they have an underlying state space
model.

An ETS model is specified by an error type (E; additive or multiplicative), a
trend type (T; additive or multiplicative, both damped or undamped, or none),
and a seasonality type (S; additive or multiplicative or none).
The following gives a very short summary, a more thorough introduction can be
found in [1]_.

Denote with :math:`\\circ_b` the trend operation (addition or
multiplication), with :math:`\\circ_d` the operation linking trend and dampening
factor :math:`\\phi` (multiplication if trend is additive, power if trend is
multiplicative), and with :math:`\\circ_s` the seasonality operation (addition
or multiplication).
Furthermore, let :math:`\\ominus` be the respective inverse operation
(subtraction or division).

With this, it is possible to formulate the ETS models as a forecast equation
and 3 smoothing equations. The former is used to forecast observations, the
latter are used to update the internal state.

.. math::

    \\hat{y}_{t|t-1} &= (l_{t-1} \\circ_b (b_{t-1}\\circ_d \\phi))\\circ_s s_{t-m}\\\\
    l_{t} &= \\alpha (y_{t} \\ominus_s s_{t-m})
             + (1 - \\alpha) (l_{t-1} \\circ_b (b_{t-1} \\circ_d \\phi))\\\\
    b_{t} &= \\beta/\\alpha (l_{t} \\ominus_b l_{t-1})
             + (1 - \\beta/\\alpha) b_{t-1}\\\\
    s_{t} &= \\gamma (y_t \\ominus_s (l_{t-1} \\circ_b (b_{t-1}\\circ_d\\phi))
             + (1 - \\gamma) s_{t-m}

The notation here follows [1]_; :math:`l_t` denotes the level at time
:math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m` is the
number of seasonal periods, and :math:`\\phi` a trend damping factor.
The parameters :math:`\\alpha, \\beta, \\gamma` are the smoothing parameters,
which are called ``smoothing_level``, ``smoothing_trend``, and
``smoothing_seasonal``, respectively.

Note that the formulation above as forecast and smoothing equation does not
distinguish different error models -- it is the same for additive and
multiplicative errors. But the different error models lead to different
likelihood models, and therefore will lead to different fit results.

The error models specify how the true values :math:`y_t` are updated. In the
additive error model,

.. math::

    y_t = \\hat{y}_{t|t-1} + e_t,

in the multiplicative error model,

.. math::

    y_t = \\hat{y}_{t|t-1}\\cdot (1 + e_t).

Using these error models, it is possible to formulate state space equations for
the ETS models:

.. math::

   y_t &= Y_t + \\eta \\cdot e_t\\\\
   l_t &= L_t + \\alpha \\cdot (M_e \\cdot L_t + \\kappa_l) \\cdot e_t\\\\
   b_t &= B_t + \\beta \\cdot (M_e \\cdot B_t + \\kappa_b) \\cdot e_t\\\\
   s_t &= S_t + \\gamma \\cdot (M_e \\cdot S_t+\\kappa_s)\\cdot e_t\\\\

with

.. math::

   B_t &= b_{t-1} \\circ_d \\phi\\\\
   L_t &= l_{t-1} \\circ_b B_t\\\\
   S_t &= s_{t-m}\\\\
   Y_t &= L_t \\circ_s S_t,

and

.. math::

   \\eta &= \\begin{cases}
               Y_t\\quad\\text{if error is multiplicative}\\\\
               1\\quad\\text{else}
           \\end{cases}\\\\
   M_e &= \\begin{cases}
               1\\quad\\text{if error is multiplicative}\\\\
               0\\quad\\text{else}
           \\end{cases}\\\\

and, when using the additive error model,

.. math::

   \\kappa_l &= \\begin{cases}
               \\frac{1}{S_t}\\quad
               \\text{if seasonality is multiplicative}\\\\
               1\\quad\\text{else}
           \\end{cases}\\\\
   \\kappa_b &= \\begin{cases}
               \\frac{\\kappa_l}{l_{t-1}}\\quad
               \\text{if trend is multiplicative}\\\\
               \\kappa_l\\quad\\text{else}
           \\end{cases}\\\\
   \\kappa_s &= \\begin{cases}
               \\frac{1}{L_t}\\quad\\text{if seasonality is multiplicative}\\\\
               1\\quad\\text{else}
           \\end{cases}

When using the multiplicative error model

.. math::

   \\kappa_l &= \\begin{cases}
               0\\quad
               \\text{if seasonality is multiplicative}\\\\
               S_t\\quad\\text{else}
           \\end{cases}\\\\
   \\kappa_b &= \\begin{cases}
               \\frac{\\kappa_l}{l_{t-1}}\\quad
               \\text{if trend is multiplicative}\\\\
               \\kappa_l + l_{t-1}\\quad\\text{else}
           \\end{cases}\\\\
   \\kappa_s &= \\begin{cases}
               0\\quad\\text{if seasonality is multiplicative}\\\\
               L_t\\quad\\text{else}
           \\end{cases}

When fitting an ETS model, the parameters :math:`\\alpha, \\beta`, \\gamma,
\\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \\ldots, s_{-m}` are
selected as maximizers of log likelihood.

References
----------
.. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
   principles and practice*, 3rd edition, OTexts: Melbourne,
   Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""
from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import array_like, bool_like, int_like, string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import _initialization_simple, _initialization_heuristic
from statsmodels.tsa.tsatools import freq_to_period

class ETSModel(base.StateSpaceMLEModel):
    """
    ETS models.

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    error : str, optional
        The error model. "add" (default) or "mul".
    trend : str or None, optional
        The trend component model. "add", "mul", or None (default).
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is
        False.
    seasonal : str, optional
        The seasonality model. "add", "mul", or None (default).
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Required if
        `seasonal` is not None.
    initialization_method : str, optional
        Method for initialization of the state space model. One of:

        * 'estimated' (default)
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_trend` and `initial_seasonal` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.  Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal_periods`.
        Only used if initialization is 'known'.
    bounds : dict or None, optional
        A dictionary with parameter names as keys and the respective bounds
        intervals as values (lists/tuples/arrays).
        The available parameter names are, depending on the model and
        initialization method:

        * "smoothing_level"
        * "smoothing_trend"
        * "smoothing_seasonal"
        * "damping_trend"
        * "initial_level"
        * "initial_trend"
        * "initial_seasonal.0", ..., "initial_seasonal.<m-1>"

        The default option is ``None``, in which case the traditional
        (nonlinear) bounds as described in [1]_ are used.

    Notes
    -----
    The ETS models are a family of time series models. They can be seen as a
    generalization of simple exponential smoothing to time series that contain
    trends and seasonalities. Additionally, they have an underlying state
    space model.

    An ETS model is specified by an error type (E; additive or multiplicative),
    a trend type (T; additive or multiplicative, both damped or undamped, or
    none), and a seasonality type (S; additive or multiplicative or none).
    The following gives a very short summary, a more thorough introduction can
    be found in [1]_.

    Denote with :math:`\\circ_b` the trend operation (addition or
    multiplication), with :math:`\\circ_d` the operation linking trend and
    dampening factor :math:`\\phi` (multiplication if trend is additive, power
    if trend is multiplicative), and with :math:`\\circ_s` the seasonality
    operation (addition or multiplication). Furthermore, let :math:`\\ominus`
    be the respective inverse operation (subtraction or division).

    With this, it is possible to formulate the ETS models as a forecast
    equation and 3 smoothing equations. The former is used to forecast
    observations, the latter are used to update the internal state.

    .. math::

        \\hat{y}_{t|t-1} &= (l_{t-1} \\circ_b (b_{t-1}\\circ_d \\phi))
                           \\circ_s s_{t-m}\\\\
        l_{t} &= \\alpha (y_{t} \\ominus_s s_{t-m})
                 + (1 - \\alpha) (l_{t-1} \\circ_b (b_{t-1} \\circ_d \\phi))\\\\
        b_{t} &= \\beta/\\alpha (l_{t} \\ominus_b l_{t-1})
                 + (1 - \\beta/\\alpha) b_{t-1}\\\\
        s_{t} &= \\gamma (y_t \\ominus_s (l_{t-1} \\circ_b (b_{t-1}\\circ_d\\phi))
                 + (1 - \\gamma) s_{t-m}

    The notation here follows [1]_; :math:`l_t` denotes the level at time
    :math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m`
    is the number of seasonal periods, and :math:`\\phi` a trend damping
    factor. The parameters :math:`\\alpha, \\beta, \\gamma` are the smoothing
    parameters, which are called ``smoothing_level``, ``smoothing_trend``, and
    ``smoothing_seasonal``, respectively.

    Note that the formulation above as forecast and smoothing equation does
    not distinguish different error models -- it is the same for additive and
    multiplicative errors. But the different error models lead to different
    likelihood models, and therefore will lead to different fit results.

    The error models specify how the true values :math:`y_t` are
    updated. In the additive error model,

    .. math::

        y_t = \\hat{y}_{t|t-1} + e_t,

    in the multiplicative error model,

    .. math::

        y_t = \\hat{y}_{t|t-1}\\cdot (1 + e_t).

    Using these error models, it is possible to formulate state space
    equations for the ETS models:

    .. math::

       y_t &= Y_t + \\eta \\cdot e_t\\\\
       l_t &= L_t + \\alpha \\cdot (M_e \\cdot L_t + \\kappa_l) \\cdot e_t\\\\
       b_t &= B_t + \\beta \\cdot (M_e \\cdot B_t + \\kappa_b) \\cdot e_t\\\\
       s_t &= S_t + \\gamma \\cdot (M_e \\cdot S_t+\\kappa_s)\\cdot e_t\\\\

    with

    .. math::

       B_t &= b_{t-1} \\circ_d \\phi\\\\
       L_t &= l_{t-1} \\circ_b B_t\\\\
       S_t &= s_{t-m}\\\\
       Y_t &= L_t \\circ_s S_t,

    and

    .. math::

       \\eta &= \\begin{cases}
                   Y_t\\quad\\text{if error is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}\\\\
       M_e &= \\begin{cases}
                   1\\quad\\text{if error is multiplicative}\\\\
                   0\\quad\\text{else}
               \\end{cases}\\\\

    and, when using the additive error model,

    .. math::

       \\kappa_l &= \\begin{cases}
                   \\frac{1}{S_t}\\quad
                   \\text{if seasonality is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_b &= \\begin{cases}
                   \\frac{\\kappa_l}{l_{t-1}}\\quad
                   \\text{if trend is multiplicative}\\\\
                   \\kappa_l\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_s &= \\begin{cases}
                   \\frac{1}{L_t}\\quad\\text{if seasonality is multiplicative}\\\\
                   1\\quad\\text{else}
               \\end{cases}

    When using the multiplicative error model

    .. math::

       \\kappa_l &= \\begin{cases}
                   0\\quad
                   \\text{if seasonality is multiplicative}\\\\
                   S_t\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_b &= \\begin{cases}
                   \\frac{\\kappa_l}{l_{t-1}}\\quad
                   \\text{if trend is multiplicative}\\\\
                   \\kappa_l + l_{t-1}\\quad\\text{else}
               \\end{cases}\\\\
       \\kappa_s &= \\begin{cases}
                   0\\quad\\text{if seasonality is multiplicative}\\\\
                   L_t\\quad\\text{else}
               \\end{cases}

    When fitting an ETS model, the parameters :math:`\\alpha, \\beta`, \\gamma,
    \\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \\ldots, s_{-m}` are
    selected as maximizers of log likelihood.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
       principles and practice*, 3rd edition, OTexts: Melbourne,
       Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
    """

    def __init__(self, endog, error='add', trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, bounds=None, dates=None, freq=None, missing='none'):
        super().__init__(endog, exog=None, dates=dates, freq=freq, missing=missing)
        options = ('add', 'mul', 'additive', 'multiplicative')
        self.error = string_like(error, 'error', options=options)[:3]
        self.trend = string_like(trend, 'trend', options=options, optional=True)
        if self.trend is not None:
            self.trend = self.trend[:3]
        self.damped_trend = bool_like(damped_trend, 'damped_trend')
        self.seasonal = string_like(seasonal, 'seasonal', options=options, optional=True)
        if self.seasonal is not None:
            self.seasonal = self.seasonal[:3]
        self.has_trend = self.trend is not None
        self.has_seasonal = self.seasonal is not None
        if self.has_seasonal:
            self.seasonal_periods = int_like(seasonal_periods, 'seasonal_periods', optional=True)
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError('seasonal_periods must be larger than 1.')
        else:
            self.seasonal_periods = 1
        if np.any(self.endog <= 0) and (self.error == 'mul' or self.trend == 'mul' or self.seasonal == 'mul'):
            raise ValueError('endog must be strictly positive when using multiplicative error, trend or seasonal components.')
        if self.damped_trend and (not self.has_trend):
            raise ValueError('Can only dampen the trend component')
        self.set_initialization_method(initialization_method, initial_level, initial_trend, initial_seasonal)
        self.set_bounds(bounds)
        if self.trend == 'add' or self.trend is None:
            if self.seasonal == 'add' or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_add_add
            else:
                self._smoothing_func = smooth._ets_smooth_add_mul
        elif self.seasonal == 'add' or self.seasonal is None:
            self._smoothing_func = smooth._ets_smooth_mul_add
        else:
            self._smoothing_func = smooth._ets_smooth_mul_mul

    def set_initialization_method(self, initialization_method, initial_level=None, initial_trend=None, initial_seasonal=None):
        """
        Sets a new initialization method for the state space model.

        Parameters
        ----------
        initialization_method : str, optional
            Method for initialization of the state space model. One of:

            * 'estimated' (default)
            * 'heuristic'
            * 'known'

            If 'known' initialization is used, then `initial_level` must be
            passed, as well as `initial_trend` and `initial_seasonal` if
            applicable.
            'heuristic' uses a heuristic based on the data to estimate initial
            level, trend, and seasonal state. 'estimated' uses the same
            heuristic as initial guesses, but then estimates the initial states
            as part of the fitting process. Default is 'estimated'.
        initial_level : float, optional
            The initial level component. Only used if initialization is
            'known'.
        initial_trend : float, optional
            The initial trend component. Only used if initialization is
            'known'.
        initial_seasonal : array_like, optional
            The initial seasonal component. An array of length
            `seasonal_periods`. Only used if initialization is 'known'.
        """
        pass

    def set_bounds(self, bounds):
        """
        Set bounds for parameter estimation.

        Parameters
        ----------
        bounds : dict or None, optional
            A dictionary with parameter names as keys and the respective bounds
            intervals as values (lists/tuples/arrays).
            The available parameter names are in ``self.param_names``.
            The default option is ``None``, in which case the traditional
            (nonlinear) bounds as described in [1]_ are used.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        pass

    @staticmethod
    def prepare_data(data):
        """
        Prepare data for use in the state space representation
        """
        pass

    def _internal_params(self, params):
        """
        Converts a parameter array passed from outside to the internally used
        full parameter array.
        """
        pass

    def _model_params(self, internal):
        """
        Converts internal parameters to model parameters
        """
        pass

    def _get_internal_states(self, states, params):
        """
        Converts a state matrix/dataframe to the (nobs, 2+m) matrix used
        internally
        """
        pass

    @property
    def _start_params(self):
        """
        Default start params in the format of external parameters.
        This should not be called directly, but by calling
        ``self.start_params``.
        """
        pass

    def _convert_and_bound_start_params(self, params):
        """
        This converts start params to internal params, sets internal-only
        parameters as bounded, sets bounds for fixed parameters, and then makes
        sure that all start parameters are within the specified bounds.
        """
        pass

    def fit(self, start_params=None, maxiter=1000, full_output=True, disp=True, callback=None, return_params=False, **kwargs):
        """
        Fit an ETS model by maximizing log-likelihood.

        Log-likelihood is a function of the model parameters :math:`\\alpha,
        \\beta, \\gamma, \\phi` (depending on the chosen model), and, if
        `initialization_method` was set to `'estimated'` in the constructor,
        also the initial states :math:`l_{-1}, b_{-1}, s_{-1}, \\ldots, s_{-m}`.

        The fit is performed using the L-BFGS algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial values for parameters that will be optimized. If this is
            ``None``, default values will be used.
            The length of this depends on the chosen model. This should contain
            the parameters in the following order, skipping parameters that do
            not exist in the chosen model.

            * `smoothing_level` (:math:`\\alpha`)
            * `smoothing_trend` (:math:`\\beta`)
            * `smoothing_seasonal` (:math:`\\gamma`)
            * `damping_trend` (:math:`\\phi`)

            If ``initialization_method`` was set to ``'estimated'`` (the
            default), additionally, the parameters

            * `initial_level` (:math:`l_{-1}`)
            * `initial_trend` (:math:`l_{-1}`)
            * `initial_seasonal.0` (:math:`s_{-1}`)
            * ...
            * `initial_seasonal.<m-1>` (:math:`s_{-m}`)

            also have to be specified.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results : ETSResults
        """
        pass

    def _loglike_internal(self, params, yhat, xhat, is_fixed=None, fixed_values=None, use_beta_star=False, use_gamma_star=False):
        """
        Log-likelihood function to be called from fit to avoid reallocation of
        memory.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m]). If there are no fixed values this must
            be in the format of internal parameters. Otherwise the fixed values
            are skipped.
        yhat : np.ndarray
            Array of size (n,) where fitted values will be written to.
        xhat : np.ndarray
            Array of size (n, _k_states_internal) where fitted states will be
            written to.
        is_fixed : np.ndarray or None
            Boolean array indicating values which are fixed during fitting.
            This must have the full length of internal parameters.
        fixed_values : np.ndarray or None
            Array of fixed values (arbitrary values for non-fixed parameters)
            This must have the full length of internal parameters.
        use_beta_star : boolean
            Whether to internally use beta_star as parameter
        use_gamma_star : boolean
            Whether to internally use gamma_star as parameter
        """
        pass

    def loglike(self, params, **kwargs):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m])

        Notes
        -----
        The log-likelihood of a exponential smoothing model is [1]_:

        .. math::

           l(\\theta, x_0|y) = - \\frac{n}{2}(\\log(2\\pi s^2) + 1)
                              - \\sum\\limits_{t=1}^n \\log(k_t)

        with

        .. math::

           s^2 = \\frac{1}{n}\\sum\\limits_{t=1}^n \\frac{(\\hat{y}_t - y_t)^2}{k_t}

        where :math:`k_t = 1` for the additive error model and :math:`k_t =
        y_t` for the multiplicative error model.

        References
        ----------
        .. [1] J. K. Ord, A. B. Koehler R. D. and Snyder (1997). Estimation and
           Prediction for a Class of Dynamic Nonlinear Statistical Models.
           *Journal of the American Statistical Association*, 92(440),
           1621-1629
        """
        pass

    def _residuals(self, yhat, data=None):
        """Calculates residuals of a prediction"""
        pass

    def _smooth(self, params):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters

        Returns
        -------
        yhat : pd.Series or np.ndarray
            Predicted values from exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.Series``, else a ``np.ndarray``.
        xhat : pd.DataFrame or np.ndarray
            Internal states of exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.DataFrame``, else a ``np.ndarray``.
        """
        pass

    def smooth(self, params, return_raw=False):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters
        return_raw : bool, optional
            Whether to return only the state space results or the full results
            object. Default is ``False``.

        Returns
        -------
        result : ETSResultsWrapper or tuple
            If ``return_raw=False``, returns a ETSResultsWrapper
            object. Otherwise a tuple of arrays or pandas objects, depending on
            the format of the endog data.
        """
        pass

    def hessian(self, params, approx_centered=False, approx_complex_step=True, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        approx_centered : bool
            Whether to use a centered scheme for finite difference
            approximation
        approx_complex_step : bool
            Whether to use complex step differentiation for approximation

        Returns
        -------
        hessian : ndarray
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.
        """
        pass

class ETSResults(base.StateSpaceMLEResults):
    """
    Results from an error, trend, seasonal (ETS) exponential smoothing model
    """

    def __init__(self, model, params, results):
        yhat, xhat = results
        self._llf = model.loglike(params)
        self._residuals = model._residuals(yhat)
        self._fittedvalues = yhat
        scale = np.mean(self._residuals ** 2)
        super().__init__(model, params, scale=scale)
        model_definition_attrs = ['short_name', 'error', 'trend', 'seasonal', 'damped_trend', 'has_trend', 'has_seasonal', 'seasonal_periods', 'initialization_method']
        for attr in model_definition_attrs:
            setattr(self, attr, getattr(model, attr))
        self.param_names = ['%s (fixed)' % name if name in self.fixed_params else name for name in self.model.param_names or []]
        internal_params = self.model._internal_params(params)
        self.states = xhat
        if self.model.use_pandas:
            states = self.states.iloc
        else:
            states = self.states
        self.initial_state = np.zeros(model._k_initial_states)
        self.level = states[:, 0]
        self.initial_level = internal_params[4]
        self.initial_state[0] = self.initial_level
        self.alpha = self.params[0]
        self.smoothing_level = self.alpha
        if self.has_trend:
            self.slope = states[:, 1]
            self.initial_trend = internal_params[5]
            self.initial_state[1] = self.initial_trend
            self.beta = self.params[1]
            self.smoothing_trend = self.beta
        if self.has_seasonal:
            self.season = states[:, self.model._seasonal_index]
            self.initial_seasonal = internal_params[6:][::-1]
            self.initial_state[self.model._seasonal_index:] = self.initial_seasonal
            self.gamma = self.params[self.model._seasonal_index]
            self.smoothing_seasonal = self.gamma
        if self.damped_trend:
            self.phi = internal_params[3]
            self.damping_trend = self.phi
        k_free_params = self.k_params - len(self.fixed_params)
        self.df_model = k_free_params + 1
        self.mean_resid = np.mean(self.resid)
        self.scale_resid = np.std(self.resid, ddof=1)
        self.standardized_forecasts_error = (self.resid - self.mean_resid) / self.scale_resid
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = 'approx'
        self._cache = {}
        self._cov_approx_complex_step = True
        self._cov_approx_centered = False
        approx_type_str = 'complex-step'
        try:
            self._rank = None
            if self.k_params == 0:
                self.cov_params_default = np.zeros((0, 0))
                self._rank = 0
                self.cov_kwds['description'] = 'No parameters estimated.'
            else:
                self.cov_params_default = self.cov_params_approx
                self.cov_kwds['description'] = descriptions['approx'].format(approx_type=approx_type_str)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = 'Covariance matrix could not be calculated: singular. information matrix.'

    @cache_readonly
    def llf(self):
        """
        log-likelihood function evaluated at the fitted params
        """
        pass

    def _get_prediction_params(self, start_idx):
        """
        Returns internal parameter representation of smoothing parameters and
        "initial" states for prediction/simulation, that is the states just
        before the first prediction/simulation step.
        """
        pass

    def _relative_forecast_variance(self, steps):
        """
        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        pass

    def simulate(self, nsimulations, anchor=None, repetitions=1, random_errors=None, random_state=None):
        """
        Random simulations using the state space formulation.

        Parameters
        ----------
        nsimulations : int
            The number of simulation steps.
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample (i.e. using the
            initial values as simulation anchor), and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
            Note: `anchor` corresponds to the observation right before the
            `start` observation in the `predict` method.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        random_errors : optional
            Specifies how the random errors should be obtained. Can be one of
            the following:

            * ``None``: Random normally distributed values with variance
              estimated from the fit errors drawn from numpy's standard
              RNG (can be seeded with the `random_state` argument). This is the
              default option.
            * A distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm``: Fits the distribution function to the fit
              errors and draws from the fitted distribution.
              Note the difference between ``scipy.stats.norm`` and
              ``scipy.stats.norm()``, the latter one is a frozen distribution
              function.
            * A frozen distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm(scale=2)``: Draws from the frozen distribution
              function.
            * A ``np.ndarray`` with shape (`nsimulations`, `repetitions`): Uses
              the given values as random errors.
            * ``"bootstrap"``: Samples the random errors from the fit errors.

        random_state : int or np.random.RandomState, optional
            A seed for the random number generator or a
            ``np.random.RandomState`` object. Only used if `random_errors` is
            ``None``. Default is ``None``.

        Returns
        -------
        sim : pd.Series, pd.DataFrame or np.ndarray
            An ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame`` of simulated
            values.
            If the original data was a ``pd.Series`` or ``pd.DataFrame``, `sim`
            will be a ``pd.Series`` if `repetitions` is 1, and a
            ``pd.DataFrame`` of shape (`nsimulations`, `repetitions`) else.
            Otherwise, if `repetitions` is 1, a ``np.ndarray`` of shape
            (`nsimulations`,) is returned, and if `repetitions` is not 1 a
            ``np.ndarray`` of shape (`nsimulations`, `repetitions`) is
            returned.
        """
        pass

    def forecast(self, steps=1):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        pass

    def _forecast(self, steps, anchor):
        """
        Dynamic prediction/forecasting
        """
        pass

    def predict(self, start=None, end=None, dynamic=False, index=None):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.

        Returns
        -------
        forecast : array_like or pd.Series.
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict,) array. If original data was a pd.Series
            or DataFrame, a pd.Series is returned.
        """
        pass

    def get_prediction(self, start=None, end=None, dynamic=False, index=None, method=None, simulate_repetitions=1000, **simulate_kwargs):
        """
        Calculates mean prediction and prediction intervals.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        method : str or None, optional
            Method to use for calculating prediction intervals. 'exact'
            (default, if available) or 'simulated'.
        simulate_repetitions : int, optional
            Number of simulation repetitions for calculating prediction
            intervals when ``method='simulated'``. Default is 1000.
        **simulate_kwargs :
            Additional arguments passed to the ``simulate`` method.

        Returns
        -------
        PredictionResults
            Predicted mean values and prediction intervals
        """
        pass

    def summary(self, alpha=0.05, start=None):
        """
        Summarize the fitted model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        pass

class ETSResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'fittedvalues': 'rows', 'level': 'rows', 'resid': 'rows', 'season': 'rows', 'slope': 'rows'}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates', 'forecast': 'dates'}
    _wrap_methods = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ETSResultsWrapper, ETSResults)

class PredictionResults:
    """
    ETS mean prediction and prediction intervals

    Parameters
    ----------
    results : ETSResults
        Model estimation results.
    start : int, str, or datetime, optional
        Zero-indexed observation number at which to start forecasting,
        i.e., the first forecast is start. Can also be a date string to
        parse or a datetime type. Default is the the zeroth observation.
    end : int, str, or datetime, optional
        Zero-indexed observation number at which to end forecasting, i.e.,
        the last forecast is end. Can also be a date string to
        parse or a datetime type. However, if the dates index does not
        have a fixed frequency, end must be an integer index if you
        want out of sample prediction. Default is the last observation in
        the sample.
    dynamic : bool, int, str, or datetime, optional
        Integer offset relative to `start` at which to begin dynamic
        prediction. Can also be an absolute date string to parse or a
        datetime type (these are not interpreted as offsets).
        Prior to this observation, true endogenous values will be used for
        prediction; starting with this observation and continuing through
        the end of prediction, forecasted endogenous values will be used
        instead.
    index : pd.Index, optional
        Optionally an index to associate the predicted results to. If None,
        an attempt is made to create an index for the predicted results
        from the model's index or model's row labels.
    method : str or None, optional
        Method to use for calculating prediction intervals. 'exact' (default,
        if available) or 'simulated'.
    simulate_repetitions : int, optional
        Number of simulation repetitions for calculating prediction intervals.
        Default is 1000.
    **simulate_kwargs :
        Additional arguments passed to the ``simulate`` method.
    """

    def __init__(self, results, start=None, end=None, dynamic=False, index=None, method=None, simulate_repetitions=1000, **simulate_kwargs):
        self.use_pandas = results.model.use_pandas
        if method is None:
            exact_available = ['ANN', 'AAN', 'AAdN', 'ANA', 'AAA', 'AAdA']
            if results.model.short_name in exact_available:
                method = 'exact'
            else:
                method = 'simulated'
        self.method = method
        start, end, start_smooth, _, anchor_dynamic, start_dynamic, end_dynamic, nsmooth, ndynamic, index = results._handle_prediction_index(start, dynamic, end, index)
        self.predicted_mean = results.predict(start=start, end=end_dynamic, dynamic=dynamic, index=index)
        self.row_labels = self.predicted_mean.index
        self.endog = np.empty(nsmooth + ndynamic) * np.nan
        if nsmooth > 0:
            self.endog[0:end - start + 1] = results.data.endog[start:end + 1]
        self.model = Bunch(data=results.model.data.__class__(endog=self.endog, predict_dates=self.row_labels))
        if self.method == 'simulated':
            sim_results = []
            if nsmooth > 1:
                if start_smooth == 0:
                    anchor = 'start'
                else:
                    anchor = start_smooth - 1
                for i in range(nsmooth):
                    sim_results.append(results.simulate(1, anchor=anchor, repetitions=simulate_repetitions, **simulate_kwargs))
                    anchor = start_smooth + i
            if ndynamic:
                sim_results.append(results.simulate(ndynamic, anchor=anchor_dynamic, repetitions=simulate_repetitions, **simulate_kwargs))
            if sim_results and isinstance(sim_results[0], pd.DataFrame):
                self.simulation_results = pd.concat(sim_results, axis=0)
            else:
                self.simulation_results = np.concatenate(sim_results, axis=0)
            self.forecast_variance = self.simulation_results.var(1)
        else:
            steps = np.ones(ndynamic + nsmooth)
            if ndynamic > 0:
                steps[start_dynamic - min(start_dynamic, start):] = range(1, ndynamic + 1)
            if start > end + 1:
                ndiscard = start - (end + 1)
                steps = steps[ndiscard:]
            self.forecast_variance = results.mse * results._relative_forecast_variance(steps)

    @property
    def var_pred_mean(self):
        """The variance of the predicted mean"""
        pass

    def pred_int(self, alpha=0.05):
        """
        Calculates prediction intervals by performing multiple simulations.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the prediction interval. Default is
            0.05, that is, a 95% prediction interval.
        """
        pass

class PredictionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'predicted_mean': 'dates', 'simulation_results': 'dates', 'endog': 'dates'}
    _wrap_attrs = wrap.union_dicts(_attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(_methods)
wrap.populate_wrapper(PredictionResultsWrapper, PredictionResults)