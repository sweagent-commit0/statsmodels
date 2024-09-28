"""
SARIMAX tools.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np

def standardize_lag_order(order, title=None):
    """
    Standardize lag order input.

    Parameters
    ----------
    order : int or array_like
        Maximum lag order (if integer) or iterable of specific lag orders.
    title : str, optional
        Description of the order (e.g. "autoregressive") to use in error
        messages.

    Returns
    -------
    order : int or list of int
        Maximum lag order if consecutive lag orders were specified, otherwise
        a list of integer lag orders.

    Notes
    -----
    It is ambiguous if order=[1] is meant to be a boolean list or
    a list of lag orders to include, but this is irrelevant because either
    interpretation gives the same result.

    Order=[0] would be ambiguous, except that 0 is not a valid lag
    order to include, so there is no harm in interpreting as a boolean
    list, in which case it is the same as order=0, which seems like
    reasonable behavior.

    Examples
    --------
    >>> standardize_lag_order(3)
    3
    >>> standardize_lag_order(np.arange(1, 4))
    3
    >>> standardize_lag_order([1, 3])
    [1, 3]
    """
    pass

def validate_basic(params, length, allow_infnan=False, title=None):
    """
    Validate parameter vector for basic correctness.

    Parameters
    ----------
    params : array_like
        Array of parameters to validate.
    length : int
        Expected length of the parameter vector.
    allow_infnan : bool, optional
            Whether or not to allow `params` to contain -np.Inf, np.Inf, and
            np.nan. Default is False.
    title : str, optional
        Description of the parameters (e.g. "autoregressive") to use in error
        messages.

    Returns
    -------
    params : ndarray
        Array of validated parameters.

    Notes
    -----
    Basic check that the parameters are numeric and that they are the right
    shape. Optionally checks for NaN / infinite values.
    """
    pass