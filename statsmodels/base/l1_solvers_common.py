"""
Holds common functions for l1 solvers.
"""
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def qc_results(params, alpha, score, qc_tol, qc_verbose=False):
    """
    Theory dictates that one of two conditions holds:
        i) abs(score[i]) == alpha[i]  and  params[i] != 0
        ii) abs(score[i]) <= alpha[i]  and  params[i] == 0
    qc_results checks to see that (ii) holds, within qc_tol

    qc_results also checks for nan or results of the wrong shape.

    Parameters
    ----------
    params : ndarray
        model parameters.  Not including the added variables x_added.
    alpha : ndarray
        regularization coefficients
    score : function
        Gradient of unregularized objective function
    qc_tol : float
        Tolerance to hold conditions (i) and (ii) to for QC check.
    qc_verbose : bool
        If true, print out a full QC report upon failure

    Returns
    -------
    passed : bool
        True if QC check passed
    qc_dict : Dictionary
        Keys are fprime, alpha, params, passed_array

    Prints
    ------
    Warning message if QC check fails.
    """
    pass

def do_trim_params(params, k_params, alpha, score, passed, trim_mode, size_trim_tol, auto_trim_tol):
    """
    Trims (set to zero) params that are zero at the theoretical minimum.
    Uses heuristics to account for the solver not actually finding the minimum.

    In all cases, if alpha[i] == 0, then do not trim the ith param.
    In all cases, do nothing with the added variables.

    Parameters
    ----------
    params : ndarray
        model parameters.  Not including added variables.
    k_params : Int
        Number of parameters
    alpha : ndarray
        regularization coefficients
    score : Function.
        score(params) should return a 1-d vector of derivatives of the
        unpenalized objective function.
    passed : bool
        True if the QC check passed
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

    Returns
    -------
    params : ndarray
        Trimmed model parameters
    trimmed : ndarray of booleans
        trimmed[i] == True if the ith parameter was trimmed.
    """
    pass