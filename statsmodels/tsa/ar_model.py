from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func, to_numpy
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, Sequence, cast
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import ArrayLike, ArrayLike1D, ArrayLike2D, Float64Array, NDArray
from statsmodels.tools.validation import array_like, bool_like, int_like, string_like
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess, DeterministicTerm, Seasonality, TimeTrend
from statsmodels.tsa.tsatools import freq_to_period, lagmat
__all__ = ['AR', 'AutoReg']
AR_DEPRECATION_WARN = "\nstatsmodels.tsa.AR has been deprecated in favor of statsmodels.tsa.AutoReg and\nstatsmodels.tsa.SARIMAX.\n\nAutoReg adds the ability to specify exogenous variables, include time trends,\nand add seasonal dummies. The AutoReg API differs from AR since the model is\ntreated as immutable, and so the entire specification including the lag\nlength must be specified when creating the model. This change is too\nsubstantial to incorporate into the existing AR api. The function\nar_select_order performs lag length selection for AutoReg models.\n\nAutoReg only estimates parameters using conditional MLE (OLS). Use SARIMAX to\nestimate ARX and related models using full MLE via the Kalman Filter.\n\nTo silence this warning and continue using AR until it is removed, use:\n\nimport warnings\nwarnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)\n"
REPEATED_FIT_ERROR = '\nModel has been fit using maxlag={0}, method={1}, ic={2}, trend={3}. These\ncannot be changed in subsequent calls to `fit`. Instead, use a new instance of\nAR.\n'

def sumofsq(x: np.ndarray, axis: int=0) -> float | np.ndarray:
    """Helper function to calculate sum of squares along first axis"""
    pass

def _get_period(data: pd.DatetimeIndex | pd.PeriodIndex, index_freq) -> int:
    """Shared helper to get period from frequenc or raise"""
    pass

class AutoReg(tsa_model.TimeSeriesModel):
    """
    Autoregressive AR-X(p) model

    Estimate an AR-X model using Conditional Maximum Likelihood (OLS).

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {None, int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
        None excludes all AR lags, and behave identically to 0.
    trend : {'n', 'c', 't', 'ct'}
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

    seasonal : bool
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    exog : array_like, optional
        Exogenous variables to include in the model. Must have the same number
        of observations as endog and should be aligned so that endog[i] is
        regressed on exog[i].
    hold_back : {None, int}
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    deterministic : DeterministicProcess
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    old_names : bool
        Flag indicating whether to use the v0.11 names or the v0.12+ names.

        .. deprecated:: 0.13.0

           old_names is deprecated and will be removed after 0.14 is
           released. You must update any code reliant on the old variable
           names to use the new names.

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Estimation of SARIMAX models using exact likelihood and the
        Kalman Filter.

    Notes
    -----
    See the notebook `Autoregressions
    <../examples/notebooks/generated/autoregressions.html>`__ for an overview.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.ar_model import AutoReg
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    >>> out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'

    Start by fitting an unrestricted Seasonal AR model

    >>> res = AutoReg(data, lags = [1, 11, 12]).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.945, HQIC: 5.970, BIC: 6.007

    An alternative used seasonal dummies

    >>> res = AutoReg(data, lags=1, seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 6.017, HQIC: 6.080, BIC: 6.175

    Finally, both the seasonal AR structure and dummies can be included

    >>> res = AutoReg(data, lags=[1, 11, 12], seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.884, HQIC: 5.959, BIC: 6.071
    """
    _y: Float64Array

    def __init__(self, endog: ArrayLike1D, lags: int | Sequence[int] | None, trend: Literal['n', 'c', 't', 'ct']='c', seasonal: bool=False, exog: ArrayLike2D | None=None, hold_back: int | None=None, period: int | None=None, missing: str='none', *, deterministic: DeterministicProcess | None=None, old_names: bool=False):
        super().__init__(endog, exog, None, None, missing=missing)
        self._trend = cast(Literal['n', 'c', 't', 'ct'], string_like(trend, 'trend', options=('n', 'c', 't', 'ct'), optional=False))
        self._seasonal = bool_like(seasonal, 'seasonal')
        self._period = int_like(period, 'period', optional=True)
        if self._period is None and self._seasonal:
            self._period = _get_period(self.data, self._index_freq)
        terms: list[DeterministicTerm] = [TimeTrend.from_string(self._trend)]
        if seasonal:
            assert isinstance(self._period, int)
            terms.append(Seasonality(self._period))
        if hasattr(self.data.orig_endog, 'index'):
            index = self.data.orig_endog.index
        else:
            index = np.arange(self.data.endog.shape[0])
        self._user_deterministic = False
        if deterministic is not None:
            if not isinstance(deterministic, DeterministicProcess):
                raise TypeError('deterministic must be a DeterministicProcess')
            self._deterministics = deterministic
            self._user_deterministic = True
        else:
            self._deterministics = DeterministicProcess(index, additional_terms=terms)
        self._exog_names: list[str] = []
        self._k_ar = 0
        self._old_names = bool_like(old_names, 'old_names', optional=False)
        if deterministic is not None and (self._trend != 'n' or self._seasonal):
            warnings.warn('When using deterministic, trend must be "n" and seasonal must be False.', SpecificationWarning, stacklevel=2)
        if self._old_names:
            warnings.warn('old_names will be removed after the 0.14 release. You should stop setting this parameter and use the new names.', FutureWarning, stacklevel=2)
        self._lags, self._hold_back = self._check_lags(lags, int_like(hold_back, 'hold_back', optional=True))
        self._setup_regressors()
        self.nobs = self._y.shape[0]
        self.data.xnames = self.exog_names

    @property
    def ar_lags(self) -> list[int] | None:
        """The autoregressive lags included in the model"""
        pass

    @property
    def hold_back(self) -> int | None:
        """The number of initial obs. excluded from the estimation sample."""
        pass

    @property
    def trend(self) -> Literal['n', 'c', 'ct', 'ctt']:
        """The trend used in the model."""
        pass

    @property
    def seasonal(self) -> bool:
        """Flag indicating that the model contains a seasonal component."""
        pass

    @property
    def deterministic(self) -> DeterministicProcess | None:
        """The deterministic used to construct the model"""
        pass

    @property
    def period(self) -> int | None:
        """The period of the seasonal component."""
        pass

    @property
    def df_model(self) -> int:
        """The model degrees of freedom."""
        pass

    @property
    def exog_names(self) -> list[str] | None:
        """Names of exogenous variables included in model"""
        pass

    def initialize(self) -> None:
        """Initialize the model (no-op)."""
        pass

    def fit(self, cov_type: str='nonrobust', cov_kwds: dict[str, Any] | None=None, use_t: bool=False) -> AutoRegResultsWrapper:
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
        AutoRegResults
            Estimation results.

        See Also
        --------
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

    def loglike(self, params: ArrayLike) -> float:
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Returns
        -------
        float
            The log-likelihood value.
        """
        pass

    def score(self, params: ArrayLike) -> np.ndarray:
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        pass

    def information(self, params: ArrayLike) -> np.ndarray:
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        ndarray
            The information matrix.
        """
        pass

    def hessian(self, params: ArrayLike) -> np.ndarray:
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        pass

    def _dynamic_predict(self, params: ArrayLike, start: int, end: int, dynamic: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
        """

        :param params:
        :param start:
        :param end:
        :param dynamic:
        :param num_oos:
        :param exog:
        :param exog_oos:
        :return:
        """
        pass

    def _static_predict(self, params: Float64Array, start: int, end: int, num_oos: int, exog: Float64Array | None, exog_oos: Float64Array | None) -> pd.Series:
        """
        Path for static predictions

        Parameters
        ----------
        params : ndarray
            The model parameters
        start : int
            Index of first observation
        end : int
            Index of last in-sample observation. Inclusive, so start:end+1
            in slice notation.
        num_oos : int
            Number of out-of-sample observations, so that the returned size is
            num_oos + (end - start + 1).
        exog : {ndarray, DataFrame}
            Array containing replacement exog values
        exog_oos :  {ndarray, DataFrame}
            Containing forecast exog values
        """
        pass

    def predict(self, params: ArrayLike, start: int | str | datetime.datetime | pd.Timestamp | None=None, end: int | str | datetime.datetime | pd.Timestamp | None=None, dynamic: bool | int=False, exog: ArrayLike2D | None=None, exog_oos: ArrayLike2D | None=None) -> pd.Series:
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
            An array containing out-of-sample values of the exogenous variable.
            Must has the same number of columns as the exog used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        pass

class AR:
    """
    The AR class has been removed and replaced with AutoReg

    See Also
    --------
    AutoReg
        The replacement for AR that improved deterministic modeling
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('AR has been removed from statsmodels and replaced with statsmodels.tsa.ar_model.AutoReg.')

class ARResults:
    """
    Removed and replaced by AutoRegResults.

    See Also
    --------
    AutoReg
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('AR and ARResults have been removed and replaced by AutoReg And AutoRegResults.')
doc = Docstring(AutoReg.predict.__doc__)
_predict_params = doc.extract_parameters(['start', 'end', 'dynamic', 'exog', 'exog_oos'], 8)

class AutoRegResults(tsa_model.TimeSeriesModelResults):
    """
    Class to hold results from fitting an AutoReg model.

    Parameters
    ----------
    model : AutoReg
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
    use_t : bool, optional
        Whether use_t was set in fit
    summary_text : str, optional
        Additional text to append to results summary
    """
    _cache: dict[str, Any] = {}

    def __init__(self, model, params, cov_params, normalized_cov_params=None, scale=1.0, use_t=False, summary_text=''):
        super().__init__(model, params, normalized_cov_params, scale)
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._use_t = use_t
        if self._ar_lags is not None:
            self._max_lag = max(self._ar_lags)
        else:
            self._max_lag = 0
        self._hold_back = self.model.hold_back
        self.cov_params_default = cov_params
        self._summary_text = summary_text

    def initialize(self, model, params, **kwargs):
        """
        Initialize (possibly re-initialize) a Results instance.

        Parameters
        ----------
        model : Model
            The model instance.
        params : ndarray
            The model parameters.
        **kwargs
            Any additional keyword arguments required to initialize the model.
        """
        pass

    @property
    def ar_lags(self):
        """The autoregressive lags included in the model"""
        pass

    @property
    def params(self):
        """The estimated parameters."""
        pass

    @property
    def df_model(self):
        """The degrees of freedom consumed by the model."""
        pass

    @property
    def df_resid(self):
        """The remaining degrees of freedom in the residuals."""
        pass

    @property
    def nobs(self):
        """
        The number of observations after adjusting for losses due to lags.
        """
        pass

    @cache_readonly
    def bse(self):
        """
        The standard errors of the estimated parameters.

        If `method` is 'cmle', then the standard errors that are returned are
        the OLS standard errors of the coefficients. If the `method` is 'mle'
        then they are computed using the numerical Hessian.
        """
        pass

    @cache_readonly
    def aic(self):
        """
        Akaike Information Criterion using Lutkepohl's definition.

        :math:`-2 llf + \\ln(nobs) (1 + df_{model})`
        """
        pass

    @cache_readonly
    def hqic(self):
        """
        Hannan-Quinn Information Criterion using Lutkepohl's definition.

        :math:`-2 llf + 2 \\ln(\\ln(nobs)) (1 + df_{model})`
        """
        pass

    @cache_readonly
    def fpe(self):
        """
        Final prediction error using Lütkepohl's definition.

        :math:`((nobs+df_{model})/(nobs-df_{model})) \\sigma^2`
        """
        pass

    @cache_readonly
    def aicc(self):
        """
        Akaike Information Criterion with small sample correction

        :math:`2.0 * df_{model} * nobs / (nobs - df_{model} - 1.0)`
        """
        pass

    @cache_readonly
    def bic(self):
        """
        Bayes Information Criterion

        :math:`-2 llf + \\ln(nobs) (1 + df_{model})`
        """
        pass

    @cache_readonly
    def resid(self):
        """
        The residuals of the model.
        """
        pass

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        pass

    @cache_readonly
    def roots(self):
        """
        The roots of the AR process.

        The roots are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0.
        Stability requires that the roots in modulus lie outside the unit
        circle.
        """
        pass

    @cache_readonly
    def arfreq(self):
        """
        Returns the frequency of the AR roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z are the
        roots.
        """
        pass

    @cache_readonly
    def fittedvalues(self):
        """
        The in-sample predicted values of the fitted AR model.

        The `k_ar` initial values are computed via the Kalman Filter if the
        model is fit by `mle`.
        """
        pass

    def test_serial_correlation(self, lags=None, model_df=None):
        """
        Ljung-Box test for residual serial correlation

        Parameters
        ----------
        lags : int
            The maximum number of lags to use in the test. Jointly tests that
            all autocorrelations up to and including lag j are zero for
            j = 1, 2, ..., lags. If None, uses min(10, nobs // 5).
        model_df : int
            The model degree of freedom to use when adjusting computing the
            test statistic to account for parameter estimation. If None, uses
            the number of AR lags included in the model.

        Returns
        -------
        output : DataFrame
            DataFrame containing three columns: the test statistic, the
            p-value of the test, and the degree of freedom used in the test.

        Notes
        -----
        Null hypothesis is no serial correlation.

        The the test degree-of-freedom is 0 or negative once accounting for
        model_df, then the test statistic's p-value is missing.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox
            Ljung-Box test for serial correlation.
        """
        pass

    def test_normality(self):
        """
        Test for normality of standardized residuals.

        Returns
        -------
        Series
            Series containing four values, the test statistic and its p-value,
            the skewness and the kurtosis.

        Notes
        -----
        Null hypothesis is normality.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera
            The Jarque-Bera test of normality.
        """
        pass

    def test_heteroskedasticity(self, lags=None):
        """
        ARCH-LM test of residual heteroskedasticity

        Parameters
        ----------
        lags : int
            The maximum number of lags to use in the test. Jointly tests that
            all squared autocorrelations up to and including lag j are zero for
            j = 1, 2, ..., lags. If None, uses lag=12*(nobs/100)^{1/4}.

        Returns
        -------
        Series
            Series containing the test statistic and its p-values.

        See Also
        --------
        statsmodels.stats.diagnostic.het_arch
            ARCH-LM test.
        statsmodels.stats.diagnostic.acorr_lm
            LM test for autocorrelation.
        """
        pass

    def diagnostic_summary(self):
        """
        Returns a summary containing standard model diagnostic tests

        Returns
        -------
        Summary
            A summary instance with panels for serial correlation tests,
            normality tests and heteroskedasticity tests.

        See Also
        --------
        test_serial_correlation
            Test models residuals for serial correlation.
        test_normality
            Test models residuals for deviations from normality.
        test_heteroskedasticity
            Test models residuals for conditional heteroskedasticity.
        """
        pass

    def get_prediction(self, start=None, end=None, dynamic=False, exog=None, exog_oos=None):
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

        Returns
        -------
        PredictionResults
            Prediction results with mean and prediction intervals
        """
        pass

    def forecast(self, steps=1, exog=None):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : {int, str, datetime}, default 1
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            steps must be an integer.
        exog : {ndarray, DataFrame}
            Exogenous values to use out-of-sample. Must have same number of
            columns as original exog data and at least `steps` rows

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.

        See Also
        --------
        AutoRegResults.predict
            In- and out-of-sample predictions
        AutoRegResults.get_prediction
            In- and out-of-sample predictions and confidence intervals
        """
        pass

    def _plot_predictions(self, predictions, start, end, alpha, in_sample, fig, figsize):
        """Shared helper for plotting predictions"""
        pass

    @Substitution(predict_params=_predict_params)
    def plot_predict(self, start=None, end=None, dynamic=False, exog=None, exog_oos=None, alpha=0.05, in_sample=True, fig=None, figsize=None):
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

    def plot_diagnostics(self, lags=10, fig=None, figsize=None):
        """
        Diagnostic plots for standardized residuals

        Parameters
        ----------
        lags : int, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residuals, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        statsmodels.graphics.tsaplots.plot_acf
        """
        pass

    def summary(self, alpha=0.05):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        smry : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        pass

    def apply(self, endog, exog=None, refit=False, fit_kwargs=None):
        """
        Apply the fitted parameters to new data unrelated to the original data

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to
        the model's original data. The new results can then be used for
        analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        refit : bool, optional
            Whether to re-fit the parameters, using the new dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`).

        Returns
        -------
        AutoRegResults
            Updated results object containing results for the new dataset.

        See Also
        --------
        AutoRegResults.append
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that are not necessarily related to the original model's `endog`
        dataset.

        Care is needed when using deterministic processes with cyclical
        components such as seasonal dummies or Fourier series. These
        deterministic components will align to the first observation
        in the data and so it is essential that any new data have the
        same initial period.

        Examples
        --------
        >>> import pandas as pd
        >>> from statsmodels.tsa.ar_model import AutoReg
        >>> index = pd.period_range(start='2000', periods=3, freq='Y')
        >>> original_observations = pd.Series([1.2, 1.5, 1.8], index=index)
        >>> mod = AutoReg(original_observations, lags=1, trend="n")
        >>> res = mod.fit()
        >>> print(res.params)
        y.L1    1.219512
        dtype: float64
        >>> print(res.fittedvalues)
        2001    1.463415
        2002    1.829268
        Freq: A-DEC, dtype: float64
        >>> print(res.forecast(1))
        2003    2.195122
        Freq: A-DEC, dtype: float64

        >>> new_index = pd.period_range(start='1980', periods=3, freq='Y')
        >>> new_observations = pd.Series([1.4, 0.3, 1.2], index=new_index)
        >>> new_res = res.apply(new_observations)
        >>> print(new_res.params)
        y.L1    1.219512
        dtype: float64
        >>> print(new_res.fittedvalues)
        1981    1.707317
        1982    0.365854
        Freq: A-DEC, dtype: float64
        >>> print(new_res.forecast(1))
        1983    1.463415
        Freq: A-DEC, dtype: float64
        """
        pass

    def append(self, endog, exog=None, refit=False, fit_kwargs=None):
        """
        Append observations to the ones used to fit the model

        Creates a new result object using the current fitted parameters
        where additional observations are appended to the data used
        to fit the model. The new results can then be used for
        analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        exog : array_like, optional
            New observations of exogenous regressors, if applicable.
        refit : bool, optional
            Whether to re-fit the parameters, using the new dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`).

        Returns
        -------
        AutoRegResults
            Updated results object containing results for the new dataset.

        See Also
        --------
        AutoRegResults.apply
        statsmodels.tsa.statespace.mlemodel.MLEResults.append

        Notes
        -----
        The endog and exog arguments to this method must be formatted in the
        same way (e.g. Pandas Series versus Numpy array) as were the endog
        and exog arrays passed to the original model.

        The endog argument to this method should consist of new observations
        that occurred directly after the last element of endog. For any other
        kind of dataset, see the apply method.

        Examples
        --------
        >>> import pandas as pd
        >>> from statsmodels.tsa.ar_model import AutoReg
        >>> index = pd.period_range(start='2000', periods=3, freq='Y')
        >>> original_observations = pd.Series([1.2, 1.4, 1.8], index=index)
        >>> mod = AutoReg(original_observations, lags=1, trend="n")
        >>> res = mod.fit()
        >>> print(res.params)
        y.L1    1.235294
        dtype: float64
        >>> print(res.fittedvalues)
        2001    1.482353
        2002    1.729412
        Freq: A-DEC, dtype: float64
        >>> print(res.forecast(1))
        2003    2.223529
        Freq: A-DEC, dtype: float64

        >>> new_index = pd.period_range(start='2003', periods=3, freq='Y')
        >>> new_observations = pd.Series([2.1, 2.4, 2.7], index=new_index)
        >>> updated_res = res.append(new_observations)
        >>> print(updated_res.params)
        y.L1    1.235294
        dtype: float64
        >>> print(updated_res.fittedvalues)
        dtype: float64
        2001    1.482353
        2002    1.729412
        2003    2.223529
        2004    2.594118
        2005    2.964706
        Freq: A-DEC, dtype: float64
        >>> print(updated_res.forecast(1))
        2006    3.335294
        Freq: A-DEC, dtype: float64
        """
        pass

class AutoRegResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(AutoRegResultsWrapper, AutoRegResults)
doc = Docstring(AutoReg.__doc__)
_auto_reg_params = doc.extract_parameters(['trend', 'seasonal', 'exog', 'hold_back', 'period', 'missing', 'old_names'], 4)

@Substitution(auto_reg_params=_auto_reg_params)
def ar_select_order(endog, maxlag, ic='bic', glob=False, trend: Literal['n', 'c', 'ct', 'ctt']='c', seasonal=False, exog=None, hold_back=None, period=None, missing='none', old_names=False):
    """
    Autoregressive AR-X(p) model order selection.

    Parameters
    ----------
    endog : array_like
         A 1-d endogenous response variable. The independent variable.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}
        The information criterion to use in the selection.
    glob : bool
        Flag indicating where to use a global search  across all combinations
        of lags.  In practice, this option is not computational feasible when
        maxlag is larger than 15 (or perhaps 20) since the global search
        requires fitting 2**maxlag models.
%(auto_reg_params)s

    Returns
    -------
    AROrderSelectionResults
        A results holder containing the model and the complete set of
        information criteria for all models fit.

    Examples
    --------
    >>> from statsmodels.tsa.ar_model import ar_select_order
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']

    Determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Determine the optimal lag structure with seasonal terms

    >>> mod = ar_select_order(data, maxlag=13, seasonal=True, period=12)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Globally determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13, glob=True)
    >>> mod.ar_lags
    array([1, 2, 9])
    """
    pass

class AROrderSelectionResults:
    """
    Results from an AR order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model: AutoReg, ics: list[tuple[int | tuple[int, ...], tuple[float, float, float]]], trend: Literal['n', 'c', 'ct', 'ctt'], seasonal: bool, period: int | None):
        self._model = model
        self._ics = ics
        self._trend = trend
        self._seasonal = seasonal
        self._period = period
        aic = sorted(ics, key=lambda r: r[1][0])
        self._aic = dict([(key, val[0]) for key, val in aic])
        bic = sorted(ics, key=lambda r: r[1][1])
        self._bic = dict([(key, val[1]) for key, val in bic])
        hqic = sorted(ics, key=lambda r: r[1][2])
        self._hqic = dict([(key, val[2]) for key, val in hqic])

    @property
    def model(self) -> AutoReg:
        """The model selected using the chosen information criterion."""
        pass

    @property
    def seasonal(self) -> bool:
        """Flag indicating if a seasonal component is included."""
        pass

    @property
    def trend(self) -> Literal['n', 'c', 'ct', 'ctt']:
        """The trend included in the model selection."""
        pass

    @property
    def period(self) -> int | None:
        """The period of the seasonal component."""
        pass

    @property
    def aic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Akaike information criterion for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        pass

    @property
    def bic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Bayesian (Schwarz) information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        pass

    @property
    def hqic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Hannan-Quinn information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        pass

    @property
    def ar_lags(self) -> list[int] | None:
        """The lags included in the selected model."""
        pass