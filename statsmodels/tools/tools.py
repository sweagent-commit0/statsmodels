"""
Utility functions models code
"""
import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like

def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    pass

def drop_missing(Y, X=None, axis=1):
    """
    Returns views on the arrays Y and X where missing observations are dropped.

    Y : array_like
    X : array_like, optional
    axis : int
        Axis along which to look for missing observations.  Default is 1, ie.,
        observations in rows.

    Returns
    -------
    Y : ndarray
        All Y where the
    X : ndarray

    Notes
    -----
    If either Y or X is 1d, it is reshaped to be 2d.
    """
    pass

def categorical(data, col=None, dictnames=False, drop=False):
    """
    Construct a dummy matrix from categorical variables

    .. deprecated:: 0.12

       Use pandas.get_dummies instead.

    Parameters
    ----------
    data : array_like
        An array, Series or DataFrame.  This can be either a 1d vector of
        the categorical variable or a 2d array with the column specifying
        the categorical variable specified by the col argument.
    col : {str, int, None}
        If data is a DataFrame col must in a column of data. If data is a
        Series, col must be either the name of the Series or None. For arrays,
        `col` can be an int that is the (zero-based) column index
        number.  `col` can only be None for a 1d array.  The default is None.
    dictnames : bool, optional
        If True, a dictionary mapping the column number to the categorical
        name is returned.  Used to have information about plain arrays.
    drop : bool
        Whether or not keep the categorical variable in the returned matrix.

    Returns
    -------
    dummy_matrix : array_like
        A matrix of dummy (indicator/binary) float variables for the
        categorical data.
    dictnames :  dict[int, str], optional
        Mapping between column numbers and categorical names.

    Notes
    -----
    This returns a dummy variable for *each* distinct variable.  If a
    a DaataFrame is provided, the names for the new variable is the
    old variable name - underscore - category name.  So if the a variable
    'vote' had answers as 'yes' or 'no' then the returned array would have to
    new variables-- 'vote_yes' and 'vote_no'.  There is currently
    no name checking.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm

    Univariate examples

    >>> import string
    >>> string_var = [string.ascii_lowercase[0:5],
    ...               string.ascii_lowercase[5:10],
    ...               string.ascii_lowercase[10:15],
    ...               string.ascii_lowercase[15:20],
    ...               string.ascii_lowercase[20:25]]
    >>> string_var *= 5
    >>> string_var = np.asarray(sorted(string_var))
    >>> design = sm.tools.categorical(string_var, drop=True)

    Or for a numerical categorical variable

    >>> instr = np.floor(np.arange(10,60, step=2)/10)
    >>> design = sm.tools.categorical(instr, drop=True)

    With a structured array

    >>> num = np.random.randn(25,2)
    >>> struct_ar = np.zeros((25,1),
    ...                      dtype=[('var1', 'f4'),('var2', 'f4'),
    ...                             ('instrument','f4'),('str_instr','a5')])
    >>> struct_ar['var1'] = num[:,0][:,None]
    >>> struct_ar['var2'] = num[:,1][:,None]
    >>> struct_ar['instrument'] = instr[:,None]
    >>> struct_ar['str_instr'] = string_var[:,None]
    >>> design = sm.tools.categorical(struct_ar, col='instrument', drop=True)

    Or

    >>> design2 = sm.tools.categorical(struct_ar, col='str_instr', drop=True)
    """
    pass

def add_constant(data, prepend=True, has_constant='skip'):
    """
    Add a column of ones to an array.

    Parameters
    ----------
    data : array_like
        A column-ordered design matrix.
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if any column has a constant value. Using 'add' will add a
        column of 1s if a constant column is present.

    Returns
    -------
    array_like
        The original values with a constant (column of ones) as the first or
        last column. Returned value type depends on input type.

    Notes
    -----
    When the input is a pandas Series or DataFrame, the added column's name
    is 'const'.
    """
    pass

def isestimable(c, d):
    """
    True if (Q, P) contrast `c` is estimable for (N, P) design `d`.

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    c : array_like
        A contrast matrix with shape (Q, P). If 1 dimensional assume shape is
        (1, P).
    d : array_like
        The design matrix, (N, P).

    Returns
    -------
    bool
        True if the contrast `c` is estimable on design `d`.

    Examples
    --------
    >>> d = np.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], d)
    False
    >>> isestimable([1, -1, 0], d)
    True
    """
    pass

def pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from numpy.
    """
    pass

def recipr(x):
    """
    Reciprocal of an array with entries less than or equal to 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
    """
    pass

def recipr0(x):
    """
    Reciprocal of an array with entries less than 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
    """
    pass

def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse.

    Parameters
    ----------
    matrix : ndarray
        The array to clean.

    Returns
    -------
    ndarray
        The cleaned array.
    """
    pass

def fullrank(x, r=None):
    """
    Return an array whose column span is the same as x.

    Parameters
    ----------
    x : ndarray
        The array to adjust, 2d.
    r : int, optional
        The rank of x. If not provided, determined by `np.linalg.matrix_rank`.

    Returns
    -------
    ndarray
        The array adjusted to have full rank.

    Notes
    -----
    If the rank of x is known it can be specified as r -- no check
    is made to ensure that this really is the rank of x.
    """
    pass

def unsqueeze(data, axis, oldshape):
    """
    Unsqueeze a collapsed array.

    Parameters
    ----------
    data : ndarray
        The data to unsqueeze.
    axis : int
        The axis to unsqueeze.
    oldshape : tuple[int]
        The original shape before the squeeze or reduce operation.

    Returns
    -------
    ndarray
        The unsqueezed array.

    Examples
    --------
    >>> from numpy import mean
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>
    """
    pass

def nan_dot(A, B):
    """
    Returns np.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : ndarray
    """
    pass

def maybe_unwrap_results(results):
    """
    Gets raw results back from wrapped results.

    Can be used in plotting functions or other post-estimation type
    routines.
    """
    pass

class Bunch(dict):
    """
    Returns a dict-like object with keys accessible via attribute lookup.

    Parameters
    ----------
    *args
        Arguments passed to dict constructor, tuples (key, value).
    **kwargs
        Keyword agument passed to dict constructor, key=value.
    """

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : ndarray, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : ndarray, DataFrame or None
        array or DataFrame with 2 dimensiona.  One dimensional arrays are
        returned as nobs by 1. None is returned if x is None.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    pass

def matrix_rank(m, tol=None, method='qr'):
    """
    Matrix rank calculation using QR or SVD

    Parameters
    ----------
    m : array_like
        A 2-d array-like object to test
    tol : float, optional
        The tolerance to use when testing the matrix rank. If not provided
        an appropriate value is selected.
    method : {"ip", "qr", "svd"}
        The method used. "ip" uses the inner-product of a normalized version
        of m and then computes the rank using NumPy's matrix_rank.
        "qr" uses a QR decomposition and is the default. "svd" defers to
        NumPy's matrix_rank.

    Returns
    -------
    int
        The rank of m.

    Notes
    -----
    When using a QR factorization, the rank is determined by the number of
    elements on the leading diagonal of the R matrix that are above tol
    in absolute value.
    """
    pass