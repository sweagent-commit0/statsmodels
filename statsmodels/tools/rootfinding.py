"""

Created on Mon Mar 18 15:48:23 2013
Author: Josef Perktold

TODO:
  - test behavior if nans or infs are encountered during the evaluation.
    now partially robust to nans, if increasing can be determined or is given.
  - rewrite core loop to use for...except instead of while.

"""
import numpy as np
from scipy import optimize
from statsmodels.tools.testing import Holder
DEBUG = False

def brentq_expanding(func, low=None, upp=None, args=(), xtol=1e-05, start_low=None, start_upp=None, increasing=None, max_it=100, maxiter_bq=100, factor=10, full_output=False):
    """find the root of a function in one variable by expanding and brentq

    Assumes function ``func`` is monotonic.

    Parameters
    ----------
    func : callable
        function for which we find the root ``x`` such that ``func(x) = 0``
    low : float or None
        lower bound for brentq
    upp : float or None
        upper bound for brentq
    args : tuple
        optional additional arguments for ``func``
    xtol : float
        parameter x tolerance given to brentq
    start_low : float (positive) or None
        starting bound for expansion with increasing ``x``. It needs to be
        positive. If None, then it is set to 1.
    start_upp : float (negative) or None
        starting bound for expansion with decreasing ``x``. It needs to be
        negative. If None, then it is set to -1.
    increasing : bool or None
        If None, then the function is evaluated at the initial bounds to
        determine wether the function is increasing or not. If increasing is
        True (False), then it is assumed that the function is monotonically
        increasing (decreasing).
    max_it : int
        maximum number of expansion steps.
    maxiter_bq : int
        maximum number of iterations of brentq.
    factor : float
        expansion factor for step of shifting the bounds interval, default is
        10.
    full_output : bool, optional
        If full_output is False, the root is returned. If full_output is True,
        the return value is (x, r), where x is the root, and r is a
        RootResults object.


    Returns
    -------
    x : float
        root of the function, value at which ``func(x) = 0``.
    info : RootResult (optional)
        returned if ``full_output`` is True.
        attributes:

         - start_bounds : starting bounds for expansion stage
         - brentq_bounds : bounds used with ``brentq``
         - iterations_expand : number of iterations in expansion stage
         - converged : True if brentq converged.
         - flag : return status, 'converged' if brentq converged
         - function_calls : number of function calls by ``brentq``
         - iterations : number of iterations in ``brentq``


    Notes
    -----
    If increasing is None, then whether the function is monotonically
    increasing or decreasing is inferred from evaluating the function at the
    initial bounds. This can fail if there is numerically no variation in the
    data in this range. In this case, using different starting bounds or
    directly specifying ``increasing`` can make it possible to move the
    expansion in the right direction.

    If

    """
    pass