from statsmodels.compat.pandas import PD_LT_2, Appender, is_numeric_dtype
from statsmodels.compat.scipy import SP_LT_19
from typing import Sequence, Union
import numpy as np
import pandas as pd
if PD_LT_2:
    from pandas.core.dtypes.common import is_categorical_dtype
from scipy import stats
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter
from statsmodels.tools.validation import array_like, bool_like, float_like, int_like
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
QUANTILES = np.array(PERCENTILES) / 100.0
MISSING = {'obs': nancount, 'mean': np.nanmean, 'std': np.nanstd, 'max': np.nanmax, 'min': np.nanmin, 'ptp': nanptp, 'var': np.nanvar, 'skew': nanskewness, 'uss': nanuss, 'kurtosis': nankurtosis, 'percentiles': nanpercentile}

def _kurtosis(a):
    """
    wrapper for scipy.stats.kurtosis that returns nan instead of raising Error

    missing options
    """
    pass

def _skew(a):
    """
    wrapper for scipy.stats.skew that returns nan instead of raising Error

    missing options
    """
    pass

def sign_test(samp, mu0=0):
    """
    Signs test

    Parameters
    ----------
    samp : array_like
        1d array. The sample for which you want to perform the sign test.
    mu0 : float
        See Notes for the definition of the sign test. mu0 is 0 by
        default, but it is common to set it to the median.

    Returns
    -------
    M
    p-value

    Notes
    -----
    The signs test returns

    M = (N(+) - N(-))/2

    where N(+) is the number of values above `mu0`, N(-) is the number of
    values below.  Values equal to `mu0` are discarded.

    The p-value for M is calculated using the binomial distribution
    and can be interpreted the same as for a t-test. The test-statistic
    is distributed Binom(min(N(+), N(-)), n_trials, .5) where n_trials
    equals N(+) + N(-).

    See Also
    --------
    scipy.stats.wilcoxon
    """
    pass
NUMERIC_STATISTICS = ('nobs', 'missing', 'mean', 'std_err', 'ci', 'std', 'iqr', 'iqr_normal', 'mad', 'mad_normal', 'coef_var', 'range', 'max', 'min', 'skew', 'kurtosis', 'jarque_bera', 'mode', 'median', 'percentiles')
CATEGORICAL_STATISTICS = ('nobs', 'missing', 'distinct', 'top', 'freq')
_additional = [stat for stat in CATEGORICAL_STATISTICS if stat not in NUMERIC_STATISTICS]
DEFAULT_STATISTICS = NUMERIC_STATISTICS + tuple(_additional)

class Description:
    """
    Extended descriptive statistics for data

    Parameters
    ----------
    data : array_like
        Data to describe. Must be convertible to a pandas DataFrame.
    stats : Sequence[str], optional
        Statistics to include. If not provided the full set of statistics is
        computed. This list may evolve across versions to reflect best
        practices. Supported options are:
        "nobs", "missing", "mean", "std_err", "ci", "ci", "std", "iqr",
        "iqr_normal", "mad", "mad_normal", "coef_var", "range", "max",
        "min", "skew", "kurtosis", "jarque_bera", "mode", "freq",
        "median", "percentiles", "distinct", "top", and "freq". See Notes for
        details.
    numeric : bool, default True
        Whether to include numeric columns in the descriptive statistics.
    categorical : bool, default True
        Whether to include categorical columns in the descriptive statistics.
    alpha : float, default 0.05
        A number between 0 and 1 representing the size used to compute the
        confidence interval, which has coverage 1 - alpha.
    use_t : bool, default False
        Use the Student's t distribution to construct confidence intervals.
    percentiles : sequence[float]
        A distinct sequence of floating point values all between 0 and 100.
        The default percentiles are 1, 5, 10, 25, 50, 75, 90, 95, 99.
    ntop : int, default 5
        The number of top categorical labels to report. Default is

    Attributes
    ----------
    numeric_statistics
        The list of supported statistics for numeric data
    categorical_statistics
        The list of supported statistics for categorical data
    default_statistics
        The default list of statistics

    See Also
    --------
    pandas.DataFrame.describe
        Basic descriptive statistics
    describe
        A simplified version that returns a DataFrame

    Notes
    -----
    The selectable statistics include:

    * "nobs" - Number of observations
    * "missing" - Number of missing observations
    * "mean" - Mean
    * "std_err" - Standard Error of the mean assuming no correlation
    * "ci" - Confidence interval with coverage (1 - alpha) using the normal or
      t. This option creates two entries in any tables: lower_ci and upper_ci.
    * "std" - Standard Deviation
    * "iqr" - Interquartile range
    * "iqr_normal" - Interquartile range relative to a Normal
    * "mad" - Mean absolute deviation
    * "mad_normal" - Mean absolute deviation relative to a Normal
    * "coef_var" - Coefficient of variation
    * "range" - Range between the maximum and the minimum
    * "max" - The maximum
    * "min" - The minimum
    * "skew" - The skewness defined as the standardized 3rd central moment
    * "kurtosis" - The kurtosis defined as the standardized 4th central moment
    * "jarque_bera" - The Jarque-Bera test statistic for normality based on
      the skewness and kurtosis. This option creates two entries, jarque_bera
      and jarque_beta_pval.
    * "mode" - The mode of the data. This option creates two entries in all tables,
      mode and mode_freq which is the empirical frequency of the modal value.
    * "median" - The median of the data.
    * "percentiles" - The percentiles. Values included depend on the input value of
      ``percentiles``.
    * "distinct" - The number of distinct categories in a categorical.
    * "top" - The mode common categories. Labeled top_n for n in 1, 2, ..., ``ntop``.
    * "freq" - The frequency of the common categories. Labeled freq_n for n in 1,
      2, ..., ``ntop``.
    """
    _int_fmt = ['nobs', 'missing', 'distinct']
    numeric_statistics = NUMERIC_STATISTICS
    categorical_statistics = CATEGORICAL_STATISTICS
    default_statistics = DEFAULT_STATISTICS

    def __init__(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], stats: Sequence[str]=None, *, numeric: bool=True, categorical: bool=True, alpha: float=0.05, use_t: bool=False, percentiles: Sequence[Union[int, float]]=PERCENTILES, ntop: bool=5):
        data_arr = data
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            data_arr = array_like(data, 'data', maxdim=2)
        if data_arr.ndim == 1:
            data = pd.Series(data)
        numeric = bool_like(numeric, 'numeric')
        categorical = bool_like(categorical, 'categorical')
        include = []
        col_types = ''
        if numeric:
            include.append(np.number)
            col_types = 'numeric'
        if categorical:
            include.append('category')
            col_types += 'and ' if col_types != '' else ''
            col_types += 'categorical'
        if not numeric and (not categorical):
            raise ValueError('At least one of numeric and categorical must be True')
        self._data = pd.DataFrame(data).select_dtypes(include)
        if self._data.shape[1] == 0:
            raise ValueError(f'Selecting {col_types} results in an empty DataFrame')
        self._is_numeric = [is_numeric_dtype(dt) for dt in self._data.dtypes]
        self._is_cat_like = [is_categorical_dtype(dt) for dt in self._data.dtypes]
        if stats is not None:
            undef = [stat for stat in stats if stat not in DEFAULT_STATISTICS]
            if undef:
                raise ValueError(f'{', '.join(undef)} are not known statistics')
        self._stats = list(DEFAULT_STATISTICS) if stats is None else list(stats)
        self._ntop = int_like(ntop, 'ntop')
        self._compute_top = 'top' in self._stats
        self._compute_freq = 'freq' in self._stats
        if self._compute_top and self._ntop <= 0 < sum(self._is_cat_like):
            raise ValueError('top must be a non-negative integer')
        replacements = {'mode': ['mode', 'mode_freq'], 'ci': ['upper_ci', 'lower_ci'], 'jarque_bera': ['jarque_bera', 'jarque_bera_pval'], 'top': [f'top_{i}' for i in range(1, self._ntop + 1)], 'freq': [f'freq_{i}' for i in range(1, self._ntop + 1)]}
        for key in replacements:
            if key in self._stats:
                idx = self._stats.index(key)
                self._stats = self._stats[:idx] + replacements[key] + self._stats[idx + 1:]
        self._percentiles = array_like(percentiles, 'percentiles', maxdim=1, dtype='d')
        self._percentiles = np.sort(self._percentiles)
        if np.unique(self._percentiles).shape[0] != self._percentiles.shape[0]:
            raise ValueError('percentiles must be distinct')
        if np.any(self._percentiles >= 100) or np.any(self._percentiles <= 0):
            raise ValueError('percentiles must be strictly between 0 and 100')
        self._alpha = float_like(alpha, 'alpha')
        if not 0 < alpha < 1:
            raise ValueError('alpha must be strictly between 0 and 1')
        self._use_t = bool_like(use_t, 'use_t')

    @cache_readonly
    def frame(self) -> pd.DataFrame:
        """
        Descriptive statistics for both numeric and categorical data

        Returns
        -------
        DataFrame
            The statistics
        """
        pass

    @cache_readonly
    def numeric(self) -> pd.DataFrame:
        """
        Descriptive statistics for numeric data

        Returns
        -------
        DataFrame
            The statistics of the numeric columns
        """
        pass

    @cache_readonly
    def categorical(self) -> pd.DataFrame:
        """
        Descriptive statistics for categorical data

        Returns
        -------
        DataFrame
            The statistics of the categorical columns
        """
        pass

    def summary(self) -> SimpleTable:
        """
        Summary table of the descriptive statistics

        Returns
        -------
        SimpleTable
            A table instance supporting export to text, csv and LaTeX
        """
        pass

    def __str__(self) -> str:
        return str(self.summary().as_text())
ds = Docstring(Description.__doc__)
ds.replace_block('Returns', Parameter(None, 'DataFrame', ['Descriptive statistics']))
ds.replace_block('Attributes', [])
ds.replace_block('See Also', [([('pandas.DataFrame.describe', None)], ['Basic descriptive statistics']), ([('Description', None)], ['Descriptive statistics class with additional output options'])])

class Describe:
    """
    Removed.
    """

    def __init__(self, dataset):
        raise NotImplementedError('Describe has been removed')