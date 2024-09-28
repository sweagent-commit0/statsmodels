from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from statsmodels.compat.python import Literal
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Hashable, Mapping, NamedTuple, Optional, Sequence, Union
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import ArrayLike1D, ArrayLike2D, Float64Array, NDArray
from statsmodels.tools.validation import array_like, bool_like, float_like, int_like
from statsmodels.tsa.ar_model import AROrderSelectionResults, AutoReg, AutoRegResults, sumofsq
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
if TYPE_CHECKING:
    import matplotlib.figure
__all__ = ['ARDL', 'ARDLResults', 'ardl_select_order', 'ARDLOrderSelectionResults', 'UECM', 'UECMResults', 'BoundsTestResult']

class BoundsTestResult(NamedTuple):
    stat: float
    crit_vals: pd.DataFrame
    p_values: pd.Series
    null: str
    alternative: str

    def __repr__(self):
        return f'{self.__class__.__name__}\nStat: {self.stat:0.5f}\nUpper P-value: {self.p_values['upper']:0.3g}\nLower P-value: {self.p_values['lower']:0.3g}\nNull: {self.null}\nAlternative: {self.alternative}\n'
_UECMOrder = Union[None, int, Dict[Hashable, Optional[int]]]
_ARDLOrder = Union[None, int, _UECMOrder, Sequence[int], Dict[Hashable, Union[int, Sequence[int], None]]]
_INT_TYPES = (int, np.integer)

class ARDL(AutoReg):
    """
    Autoregressive Distributed Lag (ARDL) Model

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If
        sequence[int], uses the ``order`` for all variables. If a dict,
        applies the lags series by series. If ``exog`` is anything other
        than a DataFrame, the keys are the column index of exog (e.g., 0,
        1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    The full specification of an ARDL is

    .. math ::

       Y_t = \\delta_0 + \\delta_1 t + \\delta_2 t^2
             + \\sum_{i=1}^{s-1} \\gamma_i I_{[(\\mod(t,s) + 1) = i]}
             + \\sum_{j=1}^p \\phi_j Y_{t-j}
             + \\sum_{l=1}^k \\sum_{m=0}^{o_l} \\beta_{l,m} X_{l, t-m}
             + Z_t \\lambda
             + \\epsilon_t

    where :math:`\\delta_\\bullet` capture trends, :math:`\\gamma_\\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See the notebook `Autoregressive Distributed Lag Models
    <../examples/notebooks/generated/autoregressive_distributed_lag.html>`__
    for an overview.

    See Also
    --------
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.ardl.UECM
        Unconstrained Error Correction Model estimation
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import ARDL
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> ARDL(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> ARDL(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> ARDL(data.lrm, [1, 3], exog, exog_lags, causal=True)

    A dictionary can also be used to pass specific lags to include.
    Sequences hold the specific lags to include, while integers are expanded
    to include [0, 1, ..., lag]. If causal is False, then the 0-th lag is
    excluded.

    >>> ARDL(lrm, [1, 3], exog, {"lry": [0, 1], "ibo": [0, 1, 3], "ide": 2})

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> ARDL(lrma, 3, exoga, {0: [0, 1], 1: [0, 1, 3], 2: 2})
    """

    def __init__(self, endog: Sequence[float] | pd.Series | ArrayLike2D, lags: int | Sequence[int] | None, exog: ArrayLike2D | None=None, order: _ARDLOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'drop', 'raise']='none') -> None:
        self._x = np.empty((0, 0))
        self._y = np.empty((0,))
        super().__init__(endog, lags, trend=trend, seasonal=seasonal, exog=exog, hold_back=hold_back, period=period, missing=missing, deterministic=deterministic, old_names=False)
        self._causal = bool_like(causal, 'causal', strict=True)
        self.data.orig_fixed = fixed
        if fixed is not None:
            fixed_arr = array_like(fixed, 'fixed', ndim=2, maxdim=2)
            if fixed_arr.shape[0] != self.data.endog.shape[0] or not np.all(np.isfinite(fixed_arr)):
                raise ValueError('fixed must be an (nobs, m) array where nobs matches the number of observations in the endog variable, and allvalues must be finite')
            if isinstance(fixed, pd.DataFrame):
                self._fixed_names = list(fixed.columns)
            else:
                self._fixed_names = [f'z.{i}' for i in range(fixed_arr.shape[1])]
            self._fixed = fixed_arr
        else:
            self._fixed = np.empty((self.data.endog.shape[0], 0))
            self._fixed_names = []
        self._blocks: dict[str, np.ndarray] = {}
        self._names: dict[str, Sequence[str]] = {}
        self._order = self._check_order(order)
        self._y, self._x = self._construct_regressors(hold_back)
        self._endog_name, self._exog_names = self._construct_variable_names()
        self.data.param_names = self.data.xnames = self._exog_names
        self.data.ynames = self._endog_name
        self._causal = True
        if self._order:
            min_lags = [min(val) for val in self._order.values()]
            self._causal = min(min_lags) > 0
        self._results_class = ARDLResults
        self._results_wrapper = ARDLResultsWrapper

    @property
    def fixed(self) -> NDArray | pd.DataFrame | None:
        """The fixed data used to construct the model"""
        pass

    @property
    def causal(self) -> bool:
        """Flag indicating that the ARDL is causal"""
        pass

    @property
    def ar_lags(self) -> list[int] | None:
        """The autoregressive lags included in the model"""
        pass

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables included in the model"""
        pass

    @property
    def ardl_order(self) -> tuple[int, ...]:
        """The order of the ARDL(p,q)"""
        pass

    def _setup_regressors(self) -> None:
        """Place holder to let AutoReg init complete"""
        pass

    @staticmethod
    def _format_exog(exog: ArrayLike2D, order: dict[Hashable, list[int]]) -> dict[Hashable, np.ndarray]:
        """Transform exogenous variables and orders to regressors"""
        pass

    def _check_order(self, order: _ARDLOrder) -> dict[Hashable, list[int]]:
        """Validate and standardize the model order"""
        pass

    def fit(self, *, cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True) -> ARDLResults:
        """
        Estimate the model parameters.

        Parameters
        ----------
        cov_type : str
            The covariance estimator to use. The most common choices are listed
            below.  Supports all covariance estimators that are available
            in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that inference should use the Student's t
            distribution that accounts for model degree of freedom.  If False,
            uses the normal distribution. If None, defers the choice to
            the cov_type. It also removes degree of freedom corrections from
            the covariance estimator when cov_type is 'nonrobust'.

        Returns
        -------
        ARDLResults
            Estimation results.

        See Also
        --------
        statsmodels.tsa.ar_model.AutoReg
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.OLS
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.RegressionResults
            See ``get_robustcov_results`` for a detailed list of available
            covariance estimators and options.

        Notes
        -----
        Use ``OLS`` to estimate model parameters and to estimate parameter
        covariance.
        """
        pass

    def _construct_regressors(self, hold_back: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        pass

    def _construct_variable_names(self):
        """Construct model variables names"""
        pass

    def _forecasting_x(self, start: int, end: int, num_oos: int, exog: ArrayLike2D | None, exog_oos: ArrayLike2D | None, fixed: ArrayLike2D | None, fixed_oos: ArrayLike2D | None) -> np.ndarray:
        """Construct exog matrix for forecasts"""
        pass

    def predict(self, params: ArrayLike1D, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None):
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous
            variables. Must have the same number of columns as the exog
            used when the model was created, and at least as many rows as
            the number of out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        pass

    @classmethod
    def from_formula(cls, formula: str, data: pd.DataFrame, lags: int | Sequence[int] | None=0, order: _ARDLOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='n', *, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'raise']='none') -> ARDL | 'UECM':
        """
        Construct an ARDL from a formula

        Parameters
        ----------
        formula : str
            Formula with form dependent ~ independent | fixed. See Examples
            below.
        data : DataFrame
            DataFrame containing the variables in the formula.
        lags : {int, list[int]}
            The number of lags to include in the model if an integer or the
            list of lag indices to include.  For example, [1, 4] will only
            include lags 1 and 4 while lags=4 will include lags 1, 2, 3,
            and 4.
        order : {int, sequence[int], dict}
            If int, uses lags 0, 1, ..., order  for all exog variables. If
            sequence[int], uses the ``order`` for all variables. If a dict,
            applies the lags series by series. If ``exog`` is anything other
            than a DataFrame, the keys are the column index of exog (e.g., 0,
            1, ...). If a DataFrame, keys are column names.
        causal : bool, optional
            Whether to include lag 0 of exog variables.  If True, only
            includes lags 1, 2, ...
        trend : {'n', 'c', 't', 'ct'}, optional
            The trend to include in the model:

            * 'n' - No trend.
            * 'c' - Constant only.
            * 't' - Time trend only.
            * 'ct' - Constant and time trend.

            The default is 'c'.

        seasonal : bool, optional
            Flag indicating whether to include seasonal dummies in the model.
            If seasonal is True and trend includes 'c', then the first period
            is excluded from the seasonal terms.
        deterministic : DeterministicProcess, optional
            A deterministic process.  If provided, trend and seasonal are
            ignored. A warning is raised if trend is not "n" and seasonal
            is not False.
        hold_back : {None, int}, optional
            Initial observations to exclude from the estimation sample.  If
            None, then hold_back is equal to the maximum lag in the model.
            Set to a non-zero value to produce comparable models with
            different lag length.  For example, to compare the fit of a model
            with lags=3 and lags=1, set hold_back=3 which ensures that both
            models are estimated using observations 3,...,nobs. hold_back
            must be >= the maximum lag in the model.
        period : {None, int}, optional
            The period of the data. Only used if seasonal is True. This
            parameter can be omitted if using a pandas object for endog
            that contains a recognized frequency.
        missing : {"none", "drop", "raise"}, optional
            Available options are 'none', 'drop', and 'raise'. If 'none', no
            NaN checking is done. If 'drop', any observations with NaNs are
            dropped. If 'raise', an error is raised. Default is 'none'.

        Returns
        -------
        ARDL
            The ARDL model instance

        Examples
        --------
        A simple ARDL using the Danish data

        >>> from statsmodels.datasets.danish_data import load
        >>> from statsmodels.tsa.api import ARDL
        >>> data = load().data
        >>> mod = ARDL.from_formula("lrm ~ ibo", data, 2, 2)

        Fixed regressors can be specified using a |

        >>> mod = ARDL.from_formula("lrm ~ ibo | ide", data, 2, 2)
        """
        pass
doc = Docstring(ARDL.predict.__doc__)
_predict_params = doc.extract_parameters(['start', 'end', 'dynamic', 'exog', 'exog_oos', 'fixed', 'fixed_oos'], 8)

class ARDLResults(AutoRegResults):
    """
    Class to hold results from fitting an ARDL model.

    Parameters
    ----------
    model : ARDL
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    use_t : bool
        Whether use_t was set in fit
    """
    _cache = {}

    def __init__(self, model: ARDL, params: np.ndarray, cov_params: np.ndarray, normalized_cov_params: Float64Array | None=None, scale: float=1.0, use_t: bool=False):
        super().__init__(model, params, normalized_cov_params, scale, use_t=use_t)
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._max_lag = 0
        if self._ar_lags:
            self._max_lag = max(self._ar_lags)
        self._hold_back = self.model.hold_back
        self.cov_params_default = cov_params

    def forecast(self, steps: int=1, exog: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None) -> np.ndarray | pd.Series:
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : {int, str, datetime}, default 1
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            steps must be an integer.
        exog : array_like, optional
            Exogenous values to use out-of-sample. Must have same number of
            columns as original exog data and at least `steps` rows
        fixed : array_like, optional
            Fixed values to use out-of-sample. Must have same number of
            columns as original fixed data and at least `steps` rows

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.

        See Also
        --------
        ARDLResults.predict
            In- and out-of-sample predictions
        ARDLResults.get_prediction
            In- and out-of-sample predictions and confidence intervals
        """
        pass

    def _lag_repr(self) -> np.ndarray:
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        pass

    def get_prediction(self, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None) -> np.ndarray | pd.Series:
        """
        Predictions and prediction intervals

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
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous variable.
            Must has the same number of columns as the exog used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        PredictionResults
            Prediction results with mean and prediction intervals
        """
        pass

    @Substitution(predict_params=_predict_params)
    def plot_predict(self, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None, alpha: float=0.05, in_sample: bool=True, fig: 'matplotlib.figure.Figure'=None, figsize: tuple[int, int] | None=None) -> 'matplotlib.figure.Figure':
        """
        Plot in- and out-of-sample predictions

        Parameters
        ----------
%(predict_params)s
        alpha : {float, None}
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float]
            Tuple containing the figure size values.

        Returns
        -------
        Figure
            Figure handle containing the plot.
        """
        pass

    def summary(self, alpha: float=0.05) -> Summary:
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        Summary
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        pass

class ARDLResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ARDLResultsWrapper, ARDLResults)

class ARDLOrderSelectionResults(AROrderSelectionResults):
    """
    Results from an ARDL order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model, ics, trend, seasonal, period):
        _ics = (((0,), (0, 0, 0)),)
        super().__init__(model, _ics, trend, seasonal, period)

        def _to_dict(d):
            return (d[0], dict(d[1:]))
        self._aic = pd.Series({v[0]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._aic.index.name = self._aic.name = 'AIC'
        self._aic = self._aic.sort_index()
        self._bic = pd.Series({v[1]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._bic.index.name = self._bic.name = 'BIC'
        self._bic = self._bic.sort_index()
        self._hqic = pd.Series({v[2]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._hqic.index.name = self._hqic.name = 'HQIC'
        self._hqic = self._hqic.sort_index()

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables in the selected model"""
        pass

def ardl_select_order(endog: ArrayLike1D | ArrayLike2D, maxlag: int, exog: ArrayLike2D, maxorder: int | dict[Hashable, int], trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, ic: Literal['aic', 'bic']='bic', glob: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'raise']='none') -> ARDLOrderSelectionResults:
    """
    ARDL order selection

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    maxlag : int
        The maximum lag to consider for the endogenous variable.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    maxorder : {int, dict}
        If int, sets a common max lag length for all exog variables. If
        a dict, then sets individual lag length. They keys are column names
        if exog is a DataFrame or column indices otherwise.
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    ic : {"aic", "bic", "hqic"}
        The information criterion to use in model selection.
    glob : bool
        Whether to consider all possible submodels of the largest model
        or only if smaller order lags must be included if larger order
        lags are.  If ``True``, the number of model considered is of the
        order 2**(maxlag + k * maxorder) assuming maxorder is an int. This
        can be very large unless k and maxorder are bot relatively small.
        If False, the number of model considered is of the order
        maxlag*maxorder**k which may also be substantial when k and maxorder
        are large.
    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Returns
    -------
    ARDLSelectionResults
        A results holder containing the selected model and the complete set
        of information criteria for all models fit.
    """
    pass
lags_descr = textwrap.wrap('The number of lags of the endogenous variable to include in the model. Must be at least 1.', 71)
lags_param = Parameter(name='lags', type='int', desc=lags_descr)
order_descr = textwrap.wrap('If int, uses lags 0, 1, ..., order  for all exog variables. If a dict, applies the lags series by series. If ``exog`` is anything other than a DataFrame, the keys are the column index of exog (e.g., 0, 1, ...). If a DataFrame, keys are column names.', 71)
order_param = Parameter(name='order', type='int, dict', desc=order_descr)
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block('Summary', 'Construct an UECM from a formula')
from_formula_doc.remove_parameters('lags')
from_formula_doc.remove_parameters('order')
from_formula_doc.insert_parameters('data', lags_param)
from_formula_doc.insert_parameters('lags', order_param)
fit_doc = Docstring(ARDL.fit.__doc__)
fit_doc.replace_block('Returns', [Parameter('', 'UECMResults', ['Estimation results.'])])
if fit_doc._ds is not None:
    see_also = fit_doc._ds['See Also']
    see_also.insert(0, ([('statsmodels.tsa.ardl.ARDL', None)], ['Autoregressive distributed lag model estimation']))
    fit_doc.replace_block('See Also', see_also)

class UECM(ARDL):
    """
    Unconstrained Error Correlation Model(UECM)

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags of the endogenous variable to include in the
        model. Must be at least 1.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If a
        dict, applies the lags series by series. If ``exog`` is anything
        other than a DataFrame, the keys are the column index of exog
        (e.g., 0, 1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    The full specification of an UECM is

    .. math ::

       \\Delta Y_t = \\delta_0 + \\delta_1 t + \\delta_2 t^2
             + \\sum_{i=1}^{s-1} \\gamma_i I_{[(\\mod(t,s) + 1) = i]}
             + \\lambda_0 Y_{t-1} + \\lambda_1 X_{1,t-1} + \\ldots
             + \\lambda_{k} X_{k,t-1}
             + \\sum_{j=1}^{p-1} \\phi_j \\Delta Y_{t-j}
             + \\sum_{l=1}^k \\sum_{m=0}^{o_l-1} \\beta_{l,m} \\Delta X_{l, t-m}
             + Z_t \\lambda
             + \\epsilon_t

    where :math:`\\delta_\\bullet` capture trends, :math:`\\gamma_\\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See Also
    --------
    statsmodels.tsa.ardl.ARDL
        Autoregressive distributed lag model estimation
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import UECM
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> UECM(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> UECM(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> UECM(data.lrm, 3, exog, exog_lags, causal=True)

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> UECM(lrma, 3, exoga, {0: 1, 1: 3, 2: 2})
    """

    def __init__(self, endog: ArrayLike1D | ArrayLike2D, lags: int | None, exog: ArrayLike2D | None=None, order: _UECMOrder=0, trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'drop', 'raise']='none') -> None:
        super().__init__(endog, lags, exog, order, trend=trend, fixed=fixed, seasonal=seasonal, causal=causal, hold_back=hold_back, period=period, missing=missing, deterministic=deterministic)
        self._results_class = UECMResults
        self._results_wrapper = UECMResultsWrapper

    def _check_lags(self, lags: int | Sequence[int] | None, hold_back: int | None) -> tuple[list[int], int]:
        """Check lags value conforms to requirement"""
        pass

    def _check_order(self, order: _ARDLOrder):
        """Check order conforms to requirement"""
        pass

    def _construct_variable_names(self):
        """Construct model variables names"""
        pass

    def _construct_regressors(self, hold_back: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        pass

    @classmethod
    def from_ardl(cls, ardl: ARDL, missing: Literal['none', 'drop', 'raise']='none'):
        """
        Construct a UECM from an ARDL model

        Parameters
        ----------
        ardl : ARDL
            The ARDL model instance
        missing : {"none", "drop", "raise"}, default "none"
            How to treat missing observations.

        Returns
        -------
        UECM
            The UECM model instance

        Notes
        -----
        The lag requirements for a UECM are stricter than for an ARDL.
        Any variable that is included in the UECM must have a lag length
        of at least 1. Additionally, the included lags must be contiguous
        starting at 0 if non-causal or 1 if causal.
        """
        pass

    def predict(self, params: ArrayLike1D, start: int | str | dt.datetime | pd.Timestamp | None=None, end: int | str | dt.datetime | pd.Timestamp | None=None, dynamic: bool=False, exog: NDArray | pd.DataFrame | None=None, exog_oos: NDArray | pd.DataFrame | None=None, fixed: NDArray | pd.DataFrame | None=None, fixed_oos: NDArray | pd.DataFrame | None=None) -> np.ndarray:
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous
            variables. Must have the same number of columns as the exog
            used when the model was created, and at least as many rows as
            the number of out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        pass

class UECMResults(ARDLResults):
    """
    Class to hold results from fitting an UECM model.

    Parameters
    ----------
    model : UECM
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    """
    _cache: dict[str, Any] = {}

    @cache_readonly
    def ci_params(self) -> np.ndarray | pd.Series:
        """Parameters of normalized cointegrating relationship"""
        pass

    @cache_readonly
    def ci_bse(self) -> np.ndarray | pd.Series:
        """Standard Errors of normalized cointegrating relationship"""
        pass

    @cache_readonly
    def ci_tvalues(self) -> np.ndarray | pd.Series:
        """T-values of normalized cointegrating relationship"""
        pass

    @cache_readonly
    def ci_pvalues(self) -> np.ndarray | pd.Series:
        """P-values of normalized cointegrating relationship"""
        pass

    def ci_cov_params(self) -> Float64Array | pd.DataFrame:
        """Covariance of normalized of cointegrating relationship"""
        pass

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        pass

    def bounds_test(self, case: Literal[1, 2, 3, 4, 5], cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True, asymptotic: bool=True, nsim: int=100000, seed: int | Sequence[int] | np.random.RandomState | np.random.Generator | None=None):
        """
        Cointegration bounds test of Pesaran, Shin, and Smith

        Parameters
        ----------
        case : {1, 2, 3, 4, 5}
            One of the cases covered in the PSS test.
        cov_type : str
            The covariance estimator to use. The asymptotic distribution of
            the PSS test has only been established in the homoskedastic case,
            which is the default.

            The most common choices are listed below.  Supports all covariance
            estimators that are available in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that small-sample corrections should be applied
            to the covariance estimator.
        asymptotic : bool
            Flag indicating whether to use asymptotic critical values which
            were computed by simulation (True, default) or to simulate a
            sample-size specific set of critical values. Tables are only
            available for up to 10 components in the cointegrating
            relationship, so if more variables are included then simulation
            is always used. The simulation computed the test statistic under
            and assumption that the residuals are homoskedastic.
        nsim : int
            Number of simulations to run when computing exact critical values.
            Only used if ``asymptotic`` is ``True``.
        seed : {None, int, sequence[int], RandomState, Generator}, optional
            Seed to use when simulating critical values. Must be provided if
            reproducible critical value and p-values are required when
            ``asymptotic`` is ``False``.

        Returns
        -------
        BoundsTestResult
            Named tuple containing ``stat``, ``crit_vals``, ``p_values``,
            ``null` and ``alternative``. The statistic is the F-type
            test statistic favored in PSS.

        Notes
        -----
        The PSS bounds test has 5 cases which test the coefficients on the
        level terms in the model

        .. math::

           \\Delta Y_{t}=\\delta_{0} + \\delta_{1}t + Z_{t-1}\\beta
                        + \\sum_{j=0}^{P}\\Delta X_{t-j}\\Gamma + \\epsilon_{t}

        where :math:`Z_{t-1}` contains both :math:`Y_{t-1}` and
        :math:`X_{t-1}`.

        The cases determine which deterministic terms are included in the
        model and which are tested as part of the test.

        Cases:

        1. No deterministic terms
        2. Constant included in both the model and the test
        3. Constant included in the model but not in the test
        4. Constant and trend included in the model, only trend included in
           the test
        5. Constant and trend included in the model, neither included in the
           test

        The test statistic is a Wald-type quadratic form test that all of the
        coefficients in :math:`\\beta` are 0 along with any included
        deterministic terms, which depends on the case. The statistic returned
        is an F-type test statistic which is the standard quadratic form test
        statistic divided by the number of restrictions.

        References
        ----------
        .. [*] Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing
           approaches to the analysis of level relationships. Journal of
           applied econometrics, 16(3), 289-326.
        """
        pass

class UECMResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(UECMResultsWrapper, UECMResults)