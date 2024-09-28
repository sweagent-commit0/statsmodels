import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
'\nElastic net regularization.\n\nRoutines for fitting regression models using elastic net\nregularization.  The elastic net minimizes the objective function\n\n-llf / nobs + alpha((1 - L1_wt) * sum(params**2) / 2 +\n    L1_wt * sum(abs(params)))\n\nThe algorithm implemented here closely follows the implementation in\nthe R glmnet package, documented here:\n\nhttp://cran.r-project.org/web/packages/glmnet/index.html\n\nand here:\n\nhttp://www.jstatsoft.org/v33/i01/paper\n\nThis routine should work for any regression model that implements\nloglike, score, and hess.\n'

def _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds):
    """
    Negative penalized log-likelihood functions.

    Returns the negative penalized log-likelihood, its derivative, and
    its Hessian.  The penalty only includes the smooth (L2) term.

    All three functions have argument signature (x, model), where
    ``x`` is a point in the parameter space and ``model`` is an
    arbitrary statsmodels regression model.
    """
    pass

def fit_elasticnet(model, method='coord_descent', maxiter=100, alpha=0.0, L1_wt=1.0, start_params=None, cnvrg_tol=1e-07, zero_tol=1e-08, refit=False, check_step=True, loglike_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Return an elastic net regularized fit to a regression model.

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``loglike``, ``score``, and
        ``hessian``.
    method : {'coord_descent'}
        Only the coordinate descent algorithm is implemented.
    maxiter : int
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array_like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in one iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    refit : bool
        If True, the model is refit using only the variables that have
        non-zero coefficients in the regularized fit.  The refitted
        model is not regularized.
    check_step : bool
        If True, confirm that the first step is an improvement and search
        further if it is not.
    loglike_kwds : dict-like or None
        Keyword arguments for the log-likelihood function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    Results
        A results object.

    Notes
    -----
    The ``elastic net`` penalty is a combination of L1 and L2
    penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then repeatedly optimize the L1 penalized version of this function
    along coordinate axes.
    """
    pass

def _opt_1d(func, grad, hess, model, start, L1_wt, tol, check_step=True):
    """
    One-dimensional helper for elastic net.

    Parameters
    ----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    model : statsmodels model
        The model being fit.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.
    check_step : bool
        If True, check that the first step is an improvement and
        use bisection if it is not.  If False, return after the
        first step regardless.

    Notes
    -----
    ``func``, ``grad``, and ``hess`` have argument signature (x,
    model), where ``x`` is a point in the parameter space and
    ``model`` is the model being fit.

    If the log-likelihood for the model is exactly quadratic, the
    global minimum is returned in one step.  Otherwise numerical
    bisection is used.

    Returns
    -------
    The argmin of the objective function.
    """
    pass

class RegularizedResults(Results):
    """
    Results for models estimated using regularization

    Parameters
    ----------
    model : Model
        The model instance used to estimate the parameters.
    params : ndarray
        The estimated (regularized) parameters.
    """

    def __init__(self, model, params):
        super(RegularizedResults, self).__init__(model, params)

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted values from the model at the estimated parameters.
        """
        pass

class RegularizedResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'params': 'columns', 'resid': 'rows', 'fittedvalues': 'rows'}
    _wrap_attrs = _attrs
wrap.populate_wrapper(RegularizedResultsWrapper, RegularizedResults)