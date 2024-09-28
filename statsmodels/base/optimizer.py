"""
Functions that are general enough to use for any model fitting. The idea is
to untie these from LikelihoodModel so that they may be re-used generally.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17

class Optimizer:

    def _fit(self, objective, gradient, start_params, fargs, kwargs, hessian=None, method='newton', maxiter=100, full_output=True, disp=True, callback=None, retall=False):
        """
        Fit function for any model with an objective function.

        Parameters
        ----------
        objective : function
            Objective function to be minimized.
        gradient : function
            The gradient of the objective function.
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        fargs : tuple
            Extra arguments passed to the objective function, i.e.
            objective(x,*args)
        kwargs : dict[str, Any]
            Extra keyword arguments passed to the objective function, i.e.
            objective(x,**kwargs)
        hessian : str, optional
            Method for computing the Hessian matrix, if applicable.
        method : str {'newton','nm','bfgs','powell','cg','ncg','basinhopping',
            'minimize'}
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, 'ncg' for Newton-
            conjugate gradient, 'basinhopping' for global basin-hopping
            solver, if available or a generic 'minimize' which is a wrapper for
            scipy.optimize.minimize. `method` determines which solver from
            scipy.optimize is used. The explicit arguments in `fit` are passed
            to the solver, with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports..
        maxiter : int
            The maximum number of iterations to perform.
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.

        Returns
        -------
        xopt : ndarray
            The solution to the objective function
        retvals : dict, None
            If `full_output` is True then this is a dictionary which holds
            information returned from the solver used. If it is False, this is
            None.
        optim_settings : dict
            A dictionary that contains the parameters passed to the solver.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for the solvers (available in Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'lbfgs'
                m : int
                    The maximum number of variable metric corrections used to
                    define the limited memory matrix. (The limited memory BFGS
                    method does not store the full hessian but uses this many
                    terms in an approximation to it.)
                pgtol : float
                    The iteration will stop when
                    ``max{|proj g_i | i = 1, ..., n} <= pgtol`` where pg_i is
                    the i-th component of the projected gradient.
                factr : float
                    The iteration stops when
                    ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
                    where eps is the machine precision, which is automatically
                    generated by the code. Typical values for factr are: 1e12
                    for low accuracy; 1e7 for moderate accuracy; 10.0 for
                    extremely high accuracy. See Notes for relationship to
                    ftol, which is exposed (instead of factr) by the
                    scipy.optimize.minimize interface to L-BFGS-B.
                maxfun : int
                    Maximum number of iterations.
                epsilon : float
                    Step size used when approx_grad is True, for numerically
                    calculating the gradient
                approx_grad : bool
                    Whether to approximate the gradient numerically (in which
                    case func returns only the function value).
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : int
                    The number of basin hopping iterations.
                niter_success : int
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : int
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                    - `args` <- `fargs`
                    - `jac` <- `score`
                    - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        pass

    def _fit_constrained(self, params):
        """
        TODO: how to add constraints?

        Something like
        sm.add_constraint(Model, func)

        or

        model_instance.add_constraint(func)
        model_instance.add_constraint("x1 + x2 = 2")
        result = model_instance.fit()
        """
        pass

def _fit_minimize(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using scipy minimize, where kwarg `min_method` defines the algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_newton(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None, ridge_factor=1e-10):
    """
    Fit using Newton-Raphson algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.
    ridge_factor : float
        Regularization factor for Hessian matrix.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_bfgs(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Broyden-Fletcher-Goldfarb-Shannon algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_lbfgs(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Limited-memory Broyden-Fletcher-Goldfarb-Shannon algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.

    Notes
    -----
    Within the mle part of statsmodels, the log likelihood function and
    its gradient with respect to the parameters do not have notationally
    consistent sign.
    """
    pass

def _fit_nm(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Nelder-Mead algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_cg(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Conjugate Gradient algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_ncg(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Newton Conjugate Gradient algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_powell(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Powell's conjugate direction algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass

def _fit_basinhopping(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Basin-hopping algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    pass