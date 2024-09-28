"""
Created on Wed May 30 15:11:09 2018

@author: josef
"""
import numpy as np
from scipy import stats

def _lm_robust(score, constraint_matrix, score_deriv_inv, cov_score, cov_params=None):
    """general formula for score/LM test

    generalized score or lagrange multiplier test for implicit constraints

    `r(params) = 0`, with gradient `R = d r / d params`

    linear constraints are given by `R params - q = 0`

    It is assumed that all arrays are evaluated at the constrained estimates.


    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    constraint_matrix R : ndarray
        Linear restriction matrix or Jacobian of nonlinear constraints
    score_deriv_inv, Ainv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be inverse of OPG or any other estimator if information
        matrix equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic
    p-value : float
        p-value of the LM test based on chisquare distribution

    Notes
    -----

    """
    pass

def score_test(self, exog_extra=None, params_constrained=None, hypothesis='joint', cov_type=None, cov_kwds=None, k_constraints=None, r_matrix=None, scale=None, observed=True):
    """score test for restrictions or for omitted variables

    Null Hypothesis : constraints are satisfied

    Alternative Hypothesis : at least one of the constraints does not hold

    This allows to specify restricted and unrestricted model properties in
    three different ways

    - fit_constrained result: model contains score and hessian function for
      the full, unrestricted model, but the parameter estimate in the results
      instance is for the restricted model. This is the case if the model
      was estimated with fit_constrained.
    - restricted model with variable addition: If exog_extra is not None, then
      it is assumed that the current model is a model with zero restrictions
      and the unrestricted model is given by adding exog_extra as additional
      explanatory variables.
    - unrestricted model with restricted parameters explicitly provided. If
      params_constrained is not None, then the model is assumed to be for the
      unrestricted model, but the provided parameters are for the restricted
      model.
      TODO: This case will currently only work for `nonrobust` cov_type,
      otherwise we will also need the restriction matrix provided by the user.


    Parameters
    ----------
    exog_extra : None or array_like
        Explanatory variables that are jointly tested for inclusion in the
        model, i.e. omitted variables.
    params_constrained : array_like
        estimated parameter of the restricted model. This can be the
        parameter estimate for the current when testing for omitted
        variables.
    hypothesis : str, 'joint' (default) or 'separate'
        If hypothesis is 'joint', then the chisquare test results for the
        joint hypothesis that all constraints hold is returned.
        If hypothesis is 'joint', then z-test results for each constraint
        is returned.
        This is currently only implemented for cov_type="nonrobust".
    cov_type : str
        Warning: only partially implemented so far, currently only "nonrobust"
        and "HC0" are supported.
        If cov_type is None, then the cov_type specified in fit for the Wald
        tests is used.
        If the cov_type argument is not None, then it will be used instead of
        the Wald cov_type given in fit.
    k_constraints : int or None
        Number of constraints that were used in the estimation of params
        restricted relative to the number of exog in the model.
        This must be provided if no exog_extra are given. If exog_extra is
        not None, then k_constraints is assumed to be zero if it is None.
    observed : bool
        If True, then the observed Hessian is used in calculating the
        covariance matrix of the score. If false then the expected
        information matrix is used. This currently only applies to GLM where
        EIM is available.
        Warning: This option might still change.

    Returns
    -------
    chi2_stat : float
        chisquare statistic for the score test
    p-value : float
        P-value of the score test based on the chisquare distribution.
    df : int
        Degrees of freedom used in the p-value calculation. This is equal
        to the number of constraints.

    Notes
    -----
    Status: experimental, several options are not implemented yet or are not
    verified yet. Currently available ptions might also still change.

    cov_type is 'nonrobust':

    The covariance matrix for the score is based on the Hessian, i.e.
    observed information matrix or optionally on the expected information
    matrix.

    cov_type is 'HC0'

    The covariance matrix of the score is the simple empirical covariance of
    score_obs without degrees of freedom correction.
    """
    pass

def _scorehess_extra(self, params=None, exog_extra=None, exog2_extra=None, hess_kwds=None):
    """Experimental helper function for variable addition score test.

    This uses score and hessian factor at the params which should be the
    params of the restricted model.

    """
    pass

def tic(results):
    """Takeuchi information criterion for misspecified models

    """
    pass

def gbic(results, gbicp=False):
    """generalized BIC for misspecified models

    References
    ----------
    Lv, Jinchi, and Jun S. Liu. 2014. "Model Selection Principles in
    Misspecified Models." Journal of the Royal Statistical Society.
    Series B (Statistical Methodology) 76 (1): 141–67.

    """
    pass