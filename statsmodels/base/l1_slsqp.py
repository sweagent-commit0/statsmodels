"""
Holds files for l1 regularization of LikelihoodModel, using
scipy.optimize.slsqp
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import statsmodels.base.l1_solvers_common as l1_solvers_common

def fit_l1_slsqp(f, score, start_params, args, kwargs, disp=False, maxiter=1000, callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using scipy.optimize.fmin_slsqp().

    Specifically:  We convert the convex but non-smooth problem

    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|

    via the transformation to the smooth, convex, constrained problem in twice
    as many variables (adding the "added variables" :math:`u_k`)

    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,

    subject to

    .. math:: -u_k \\leq \\beta_k \\leq u_k.

    Parameters
    ----------
    All the usual parameters from LikelhoodModel.fit
    alpha : non-negative scalar or numpy array (same size as parameters)
        The weight multiplying the l1 penalty term
    trim_mode : 'auto, 'size', or 'off'
        If not 'off', trim (set to zero) parameters that would have been zero
            if the solver reached the theoretical minimum.
        If 'auto', trim params using the Theory above.
        If 'size', trim params if they have very small absolute value
    size_trim_tol : float or 'auto' (default = 'auto')
        For use when trim_mode === 'size'
    auto_trim_tol : float
        For sue when trim_mode == 'auto'.  Use
    qc_tol : float
        Print warning and do not allow auto trim when (ii) in "Theory" (above)
        is violated by this much.
    qc_verbose : bool
        If true, print out a full QC report upon failure
    acc : float (default 1e-6)
        Requested accuracy as used by slsqp
    """
    pass

def _objective_func(f, x_full, k_params, alpha, *args):
    """
    The regularized objective function
    """
    pass

def _fprime(score, x_full, k_params, alpha):
    """
    The regularized derivative
    """
    pass

def _f_ieqcons(x_full, k_params):
    """
    The inequality constraints.
    """
    pass

def _fprime_ieqcons(x_full, k_params):
    """
    Derivative of the inequality constraints
    """
    pass