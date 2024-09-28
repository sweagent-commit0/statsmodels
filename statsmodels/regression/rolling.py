"""
Rolling OLS and WLS

Implements an efficient rolling estimator that avoids repeated matrix
multiplication.

Copyright (c) 2019 Kevin Sheppard
License: 3-clause BSD
"""
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import Appender, Substitution, cache_readonly, call_cached_func, get_cached_doc
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import RegressionModel, RegressionResults
from statsmodels.tools.validation import array_like, int_like, string_like
RollingStore = namedtuple('RollingStore', ['params', 'ssr', 'llf', 'nobs', 's2', 'xpxi', 'xeex', 'centered_tss', 'uncentered_tss'])
common_params = '\n'.join(map(strip4, model._model_params_doc.split('\n')))
window_parameters = 'window : int\n    Length of the rolling window. Must be strictly larger than the number\n    of variables in the model.\n'
weight_parameters = '\nweights : array_like, optional\n    A 1d array of weights.  If you supply 1/W then the variables are\n    pre- multiplied by 1/sqrt(W).  If no weights are supplied the\n    default value is 1 and WLS results are the same as OLS.\n'
_missing_param_doc = 'min_nobs : {int, None}\n    Minimum number of observations required to estimate a model when\n    data are missing.  If None, the minimum depends on the number of\n    regressors in the model. Must be smaller than window.\nmissing : str, default "drop"\n    Available options are "drop", "skip" and "raise". If "drop", any\n    observations with nans are dropped and the estimates are computed using\n    only the non-missing values in each window. If \'skip\' blocks containing\n    missing values are skipped and the corresponding results contains NaN.\n    If \'raise\', an error is raised. Default is \'drop\'.\nexpanding : bool, default False\n    If True, then the initial observations after min_nobs are filled using\n    an expanding scheme until ``window`` observations are available, after\n    which rolling is used.\n'
extra_base = _missing_param_doc
extra_parameters = window_parameters + weight_parameters + extra_base
_doc = '\nRolling %(model_type)s Least Squares\n\n%(parameters)s\n%(extra_parameters)s\n\nSee Also\n--------\nstatsmodels.regression.linear_model.%(model)s\n    %(model)s estimation and parameter testing.\n\nNotes\n-----\nTested against %(model)s for accuracy.\n\nResults may differ from %(model)s applied to windows of data if this\nmodel contains an implicit constant (i.e., includes dummies for all\ncategories) rather than an explicit constant (e.g., a column of 1s).\n\nExamples\n--------\n>>> from statsmodels.regression.rolling import Rolling%(model)s\n>>> from statsmodels.datasets import longley\n>>> data = longley.load()\n>>> exog = add_constant(data.exog, prepend=False)\n>>> mod = Rolling%(model)s(data.endog, exog)\n>>> rolling_res = mod.fit(reset=50)\n\nUse params_only to skip all calculations except parameter estimation\n\n>>> rolling_params = mod.fit(params_only=True)\n\nUse expanding and min_nobs to fill the initial results using an\nexpanding scheme until window observation, and the roll.\n\n>>> mod = Rolling%(model)s(data.endog, exog, window=60, min_nobs=12,\n... expanding=True)\n>>> rolling_res = mod.fit()\n'

@Substitution(model_type='Weighted', model='WLS', parameters=common_params, extra_parameters=extra_parameters)
@Appender(_doc)
class RollingWLS:

    def __init__(self, endog, exog, window=None, *, weights=None, min_nobs=None, missing='drop', expanding=False):
        missing = string_like(missing, 'missing', options=('drop', 'raise', 'skip'))
        temp_msng = 'drop' if missing != 'raise' else 'raise'
        Model.__init__(self, endog, exog, missing=temp_msng, hasconst=None)
        k_const = self.k_constant
        const_idx = self.data.const_idx
        Model.__init__(self, endog, exog, missing='none', hasconst=False)
        self.k_constant = k_const
        self.data.const_idx = const_idx
        self._y = array_like(endog, 'endog')
        nobs = self._y.shape[0]
        self._x = array_like(exog, 'endog', ndim=2, shape=(nobs, None))
        window = int_like(window, 'window', optional=True)
        weights = array_like(weights, 'weights', optional=True, shape=(nobs,))
        self._window = window if window is not None else self._y.shape[0]
        self._weighted = weights is not None
        self._weights = np.ones(nobs) if weights is None else weights
        w12 = np.sqrt(self._weights)
        self._wy = w12 * self._y
        self._wx = w12[:, None] * self._x
        min_nobs = int_like(min_nobs, 'min_nobs', optional=True)
        self._min_nobs = min_nobs if min_nobs is not None else self._x.shape[1]
        if self._min_nobs < self._x.shape[1] or self._min_nobs > self._window:
            raise ValueError('min_nobs must be larger than the number of regressors in the model and less than window')
        self._expanding = expanding
        self._is_nan = np.zeros_like(self._y, dtype=bool)
        self._has_nan = self._find_nans()
        self.const_idx = self.data.const_idx
        self._skip_missing = missing == 'skip'

    def _reset(self, idx):
        """Compute xpx and xpy using a single dot product"""
        pass

    def fit(self, method='inv', cov_type='nonrobust', cov_kwds=None, reset=None, use_t=False, params_only=False):
        """
        Estimate model parameters.

        Parameters
        ----------
        method : {'inv', 'lstsq', 'pinv'}
            Method to use when computing the the model parameters.

            * 'inv' - use moving windows inner-products and matrix inversion.
              This method is the fastest, but may be less accurate than the
              other methods.
            * 'lstsq' - Use numpy.linalg.lstsq
            * 'pinv' - Use numpy.linalg.pinv. This method matches the default
              estimator in non-moving regression estimators.
        cov_type : {'nonrobust', 'HCCM', 'HC0'}
            Covariance estimator:

            * nonrobust - The classic OLS covariance estimator
            * HCCM, HC0 - White heteroskedasticity robust covariance
        cov_kwds : dict
            Unused
        reset : int, optional
            Interval to recompute the moving window inner products used to
            estimate the model parameters. Smaller values improve accuracy,
            although in practice this setting is not required to be set.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.
        params_only : bool, optional
            Flag indicating that only parameters should be computed. Avoids
            calculating all other statistics or performing inference.

        Returns
        -------
        RollingRegressionResults
            Estimation results where all pre-sample values are nan-filled.
        """
        pass
extra_parameters = window_parameters + extra_base

@Substitution(model_type='Ordinary', model='OLS', parameters=common_params, extra_parameters=extra_parameters)
@Appender(_doc)
class RollingOLS(RollingWLS):

    def __init__(self, endog, exog, window=None, *, min_nobs=None, missing='drop', expanding=False):
        super().__init__(endog, exog, window, weights=None, min_nobs=min_nobs, missing=missing, expanding=expanding)

class RollingRegressionResults:
    """
    Results from rolling regressions

    Parameters
    ----------
    model : RollingWLS
        Model instance
    store : RollingStore
        Container for raw moving window results
    k_constant : bool
        Flag indicating that the model contains a constant
    use_t : bool
        Flag indicating to use the Student's t distribution when computing
        p-values.
    cov_type : str
        Name of covariance estimator
    """
    _data_in_cache = tuple()

    def __init__(self, model, store: RollingStore, k_constant, use_t, cov_type):
        self.model = model
        self._params = store.params
        self._ssr = store.ssr
        self._llf = store.llf
        self._nobs = store.nobs
        self._s2 = store.s2
        self._xpxi = store.xpxi
        self._xepxe = store.xeex
        self._centered_tss = store.centered_tss
        self._uncentered_tss = store.uncentered_tss
        self._k_constant = k_constant
        self._nvar = self._xpxi.shape[-1]
        if use_t is None:
            use_t = cov_type == 'nonrobust'
        self._use_t = use_t
        self._cov_type = cov_type
        self._use_pandas = self.model.data.row_labels is not None
        self._data_attr = []
        self._cache = {}

    def _wrap(self, val):
        """Wrap output as pandas Series or DataFrames as needed"""
        pass

    @cache_readonly
    def params(self):
        """Estimated model parameters"""
        pass

    @cache_readonly
    def k_constant(self):
        """Flag indicating whether the model contains a constant"""
        pass

    def cov_params(self):
        """
        Estimated parameter covariance

        Returns
        -------
        array_like
            The estimated model covariances. If the original input is a numpy
            array, the returned covariance is a 3-d array with shape
            (nobs, nvar, nvar). If the original inputs are pandas types, then
            the returned covariance is a DataFrame with a MultiIndex with
            key (observation, variable), so that the covariance for
            observation with index i is cov.loc[i].
        """
        pass

    @property
    def cov_type(self):
        """Name of covariance estimator"""
        pass
    remove_data = LikelihoodModelResults.remove_data

    def plot_recursive_coefficient(self, variables=None, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the recursively estimated coefficients on a given variable

        Parameters
        ----------
        variables : {int, str, Iterable[int], Iterable[str], None}, optional
            Integer index or string name of the variables whose coefficients
            to plot. Can also be an iterable of integers or strings. Default
            plots all coefficients.
        alpha : float, optional
            The confidence intervals for the coefficient are (1 - alpha)%. Set
            to None to exclude confidence intervals.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Returns
        -------
        Figure
            The matplotlib Figure object.
        """
        pass