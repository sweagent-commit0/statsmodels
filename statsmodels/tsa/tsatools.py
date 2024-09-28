from __future__ import annotations
from statsmodels.compat.python import Literal, lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import array_like, bool_like, int_like, string_like
__all__ = ['lagmat', 'lagmat2ds', 'add_trend', 'duplication_matrix', 'elimination_matrix', 'commutation_matrix', 'vec', 'vech', 'unvec', 'unvech', 'freq_to_period']

def add_trend(x, trend='c', prepend=False, has_constant='skip'):
    """
    Add a trend and/or constant to an array.

    Parameters
    ----------
    x : array_like
        Original array of data.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant column already
        exists in x. 'raise' will raise an error. 'add' will add a column of
        1s. 'skip' will return the data without change. 'skip' is the default.

    Returns
    -------
    array_like
        The original data with the additional trend columns.  If x is a
        pandas Series or DataFrame, then the trend column names are 'const',
        'trend' and 'trend_squared'.

    See Also
    --------
    statsmodels.tools.tools.add_constant
        Add a constant column to an array.

    Notes
    -----
    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
    no checking for an existing trend.
    """
    pass

def add_lag(x, col=None, lags=1, drop=False, insert=True):
    """
    Returns an array with lags included given an array.

    Parameters
    ----------
    x : array_like
        An array or NumPy ndarray subclass. Can be either a 1d or 2d array with
        observations in columns.
    col : int or None
        `col` can be an int of the zero-based column index. If it's a
        1d array `col` can be None.
    lags : int
        The number of lags desired.
    drop : bool
        Whether to keep the contemporaneous variable for the data.
    insert : bool or int
        If True, inserts the lagged values after `col`. If False, appends
        the data. If int inserts the lags at int.

    Returns
    -------
    array : ndarray
        Array with lags

    Examples
    --------

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load()
    >>> data = data.data[['year','quarter','realgdp','cpi']]
    >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)

    Notes
    -----
    Trims the array both forward and backward, so that the array returned
    so that the length of the returned array is len(`X`) - lags. The lags are
    returned in increasing order, ie., t-1,t-2,...,t-lags
    """
    pass

def detrend(x, order=1, axis=0):
    """
    Detrend an array with a trend of given order along axis 0 or 1.

    Parameters
    ----------
    x : array_like, 1d or 2d
        Data, if 2d, then each row or column is independently detrended with
        the same trendorder, but independent trend estimates.
    order : int
        The polynomial order of the trend, zero is constant, one is
        linear trend, two is quadratic trend.
    axis : int
        Axis can be either 0, observations by rows, or 1, observations by
        columns.

    Returns
    -------
    ndarray
        The detrended series is the residual of the linear regression of the
        data on the trend of given order.
    """
    pass

def lagmat(x, maxlag: int, trim: Literal['forward', 'backward', 'both', 'none']='forward', original: Literal['ex', 'sep', 'in']='ex', use_pandas: bool=False) -> NDArray | DataFrame | tuple[NDArray, NDArray] | tuple[DataFrame, DataFrame]:
    """
    Create 2d array of lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend
    index values.

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])
    """
    pass

def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward', use_pandas=False):
    """
    Generate lagmatrix for 2d array, columns arranged by variables.

    Parameters
    ----------
    x : array_like
        Data, 2d. Observations in rows and variables in columns.
    maxlag0 : int
        The first variable all lags from zero to maxlag are included.
    maxlagex : {None, int}
        The max lag for all other variables all lags from zero to maxlag are
        included.
    dropex : int
        Exclude first dropex lags from other variables. For all variables,
        except the first, lags from dropex to maxlagex are included.
    trim : str
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none' : no trimming of observations.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    ndarray
        The array with lagged observations, columns ordered by variable.

    Notes
    -----
    Inefficient implementation for unequal lags, implemented for convenience.
    """
    pass

def duplication_matrix(n):
    """
    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for
    symmetric matrix S

    Returns
    -------
    D_n : ndarray
    """
    pass

def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M

    Parameters
    ----------

    Returns
    -------
    """
    pass

def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p : int
    q : int

    Returns
    -------
    K : ndarray (pq x pq)
    """
    pass

def _ar_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array_like
        The AR coefficients

    Reference
    ---------
    Jones(1980)
    """
    pass

def _ar_invtransparams(params):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array_like
        The transformed AR coefficients
    """
    pass

def _ma_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : ndarray
        The ma coeffecients of an (AR)MA model.

    Reference
    ---------
    Jones(1980)
    """
    pass

def _ma_invtransparams(macoefs):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : ndarray
        The transformed MA coefficients
    """
    pass

def unintegrate_levels(x, d):
    """
    Returns the successive differences needed to unintegrate the series.

    Parameters
    ----------
    x : array_like
        The original series
    d : int
        The number of differences of the differenced series.

    Returns
    -------
    y : array_like
        The increasing differences from 0 to d-1 of the first d elements
        of x.

    See Also
    --------
    unintegrate
    """
    pass

def unintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array_like
        The n-th differenced series
    levels : list
        A list of the first-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array_like
        The original series de-differenced

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = unintegrate_levels(x, 2)
    >>> levels
    array([ 1.,  2.])
    >>> unintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    pass

def freq_to_period(freq: str | offsets.DateOffset) -> int:
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    pass