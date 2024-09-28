"""numerical differentiation function, gradient, Jacobian, and Hessian

Author : josef-pkt
License : BSD

Notes
-----
These are simple forward differentiation, so that we have them available
without dependencies.

* Jacobian should be faster than numdifftools because it does not use loop over
  observations.
* numerical precision will vary and depend on the choice of stepsizes
"""
import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
EPS = np.finfo(float).eps
_hessian_docs = '\n    Calculate Hessian with finite difference derivative approximation\n\n    Parameters\n    ----------\n    x : array_like\n       value at which function derivative is evaluated\n    f : function\n       function of one array f(x, `*args`, `**kwargs`)\n    epsilon : float or array_like, optional\n       Stepsize used, if None, then stepsize is automatically chosen\n       according to EPS**(1/%(scale)s)*x.\n    args : tuple\n        Arguments for function `f`.\n    kwargs : dict\n        Keyword arguments for function `f`.\n    %(extra_params)s\n\n    Returns\n    -------\n    hess : ndarray\n       array of partial second derivatives, Hessian\n    %(extra_returns)s\n\n    Notes\n    -----\n    Equation (%(equation_number)s) in Ridout. Computes the Hessian as::\n\n      %(equation)s\n\n    where e[j] is a vector with element j == 1 and the rest are zero and\n    d[i] is epsilon[i].\n\n    References\n    ----------:\n\n    Ridout, M.S. (2009) Statistical applications of the complex-step method\n        of numerical differentiation. The American Statistician, 63, 66-74\n'

def approx_fprime(x, f, epsilon=None, args=(), kwargs={}, centered=False):
    """
    Gradient of function, or Jacobian if function f returns 1d array

    Parameters
    ----------
    x : ndarray
        parameters at which the derivative is evaluated
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.

    Returns
    -------
    grad : ndarray
        gradient or Jacobian

    Notes
    -----
    If f returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by f (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """
    pass

def _approx_fprime_scalar(x, f, epsilon=None, args=(), kwargs={}, centered=False):
    """
    Gradient of function vectorized for scalar parameter.

    This assumes that the function ``f`` is vectorized for a scalar parameter.
    The function value ``f(x)`` has then the same shape as the input ``x``.
    The derivative returned by this function also has the same shape as ``x``.

    Parameters
    ----------
    x : ndarray
        Parameters at which the derivative is evaluated.
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.

    Returns
    -------
    grad : ndarray
        Array of derivatives, gradient evaluated at parameters ``x``.
    """
    pass

def approx_fprime_cs(x, f, epsilon=None, args=(), kwargs={}):
    """
    Calculate gradient or Jacobian with complex step derivative approximation

    Parameters
    ----------
    x : ndarray
        parameters at which the derivative is evaluated
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. Optimal step-size is
        EPS*x. See note.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.

    Returns
    -------
    partials : ndarray
       array of partial derivatives, Gradient or Jacobian

    Notes
    -----
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction.
    """
    pass

def _approx_fprime_cs_scalar(x, f, epsilon=None, args=(), kwargs={}):
    """
    Calculate gradient for scalar parameter with complex step derivatives.

    This assumes that the function ``f`` is vectorized for a scalar parameter.
    The function value ``f(x)`` has then the same shape as the input ``x``.
    The derivative returned by this function also has the same shape as ``x``.

    Parameters
    ----------
    x : ndarray
        Parameters at which the derivative is evaluated.
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array.
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. Optimal step-size is
        EPS*x. See note.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.

    Returns
    -------
    partials : ndarray
       Array of derivatives, gradient evaluated for parameters ``x``.

    Notes
    -----
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction.
    """
    pass

def approx_hess_cs(x, f, epsilon=None, args=(), kwargs={}):
    """Calculate Hessian with complex-step derivative approximation

    Parameters
    ----------
    x : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x)
    epsilon : float
       stepsize, if None, then stepsize is automatically chosen

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian

    Notes
    -----
    based on equation 10 in
    M. S. RIDOUT: Statistical Applications of the Complex-step Method
    of Numerical Differentiation, University of Kent, Canterbury, Kent, U.K.

    The stepsize is the same for the complex and the finite difference part.
    """
    pass
approx_hess = approx_hess3
approx_hess.__doc__ += '\n    This is an alias for approx_hess3'