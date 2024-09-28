from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly

def _check_margeff_args(at, method):
    """
    Checks valid options for margeff
    """
    pass

def _check_discrete_args(at, method):
    """
    Checks the arguments for margeff if the exogenous variables are discrete.
    """
    pass

def _get_const_index(exog):
    """
    Returns a boolean array of non-constant column indices in exog and
    an scalar array of where the constant is or None
    """
    pass

def _isdummy(X):
    """
    Given an array X, returns the column indices for the dummy variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _isdummy(X)
    >>> ind
    array([0, 3, 4])
    """
    pass

def _iscount(X):
    """
    Given an array X, returns the column indices for count variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 10, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _iscount(X)
    >>> ind
    array([0, 3, 4])
    """
    pass

def _get_count_effects(effects, exog, count_ind, method, model, params):
    """
    If there's a count variable, the predicted difference is taken by
    subtracting one and adding one to exog then averaging the difference
    """
    pass

def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    If there's a dummy variable, the predicted difference is taken at
    0 and 1
    """
    pass

def _margeff_cov_params_dummy(model, cov_margins, params, exog, dummy_ind, method, J):
    """
    Returns the Jacobian for discrete regressors for use in margeff_cov_params.

    For discrete regressors the marginal effect is

    \\Delta F = F(XB) | d = 1 - F(XB) | d = 0

    The row of the Jacobian for this variable is given by

    f(XB)*X | d = 1 - f(XB)*X | d = 0

    Where F is the default prediction of the model.
    """
    pass

def _margeff_cov_params_count(model, cov_margins, params, exog, count_ind, method, J):
    """
    Returns the Jacobian for discrete regressors for use in margeff_cov_params.

    For discrete regressors the marginal effect is

    \\Delta F = F(XB) | d += 1 - F(XB) | d -= 1

    The row of the Jacobian for this variable is given by

    (f(XB)*X | d += 1 - f(XB)*X | d -= 1) / 2

    where F is the default prediction for the model.
    """
    pass

def margeff_cov_params(model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J):
    """
    Computes the variance-covariance of marginal effects by the delta method.

    Parameters
    ----------
    model : model instance
        The model that returned the fitted results. Its pdf method is used
        for computing the Jacobian of discrete variables in dummy_ind and
        count_ind
    params : array_like
        estimated model parameters
    exog : array_like
        exogenous variables at which to calculate the derivative
    cov_params : array_like
        The variance-covariance of the parameters
    at : str
       Options are:

        - 'overall', The average of the marginal effects at each
          observation.
        - 'mean', The marginal effects at the mean of each regressor.
        - 'median', The marginal effects at the median of each regressor.
        - 'zero', The marginal effects at zero for each regressor.
        - 'all', The marginal effects at each observation.

        Only overall has any effect here.you

    derivative : function or array_like
        If a function, it returns the marginal effects of the model with
        respect to the exogenous variables evaluated at exog. Expected to be
        called derivative(params, exog). This will be numerically
        differentiated. Otherwise, it can be the Jacobian of the marginal
        effects with respect to the parameters.
    dummy_ind : array_like
        Indices of the columns of exog that contain dummy variables
    count_ind : array_like
        Indices of the columns of exog that contain count variables

    Notes
    -----
    For continuous regressors, the variance-covariance is given by

    Asy. Var[MargEff] = [d margeff / d params] V [d margeff / d params]'

    where V is the parameter variance-covariance.

    The outer Jacobians are computed via numerical differentiation if
    derivative is a function.
    """
    pass

def margeff_cov_with_se(model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J):
    """
    See margeff_cov_params.

    Same function but returns both the covariance of the marginal effects
    and their standard errors.
    """
    pass
_transform_names = dict(dydx='dy/dx', eyex='d(lny)/d(lnx)', dyex='dy/d(lnx)', eydx='d(lny)/dx')

class Margins:
    """
    Mostly a do nothing class. Lays out the methods expected of a sub-class.

    This is just a sketch of what we may want out of a general margins class.
    I (SS) need to look at details of other models.
    """

    def __init__(self, results, get_margeff, derivative, dist=None, margeff_args=()):
        self._cache = {}
        self.results = results
        self.dist = dist
        self.get_margeff(margeff_args)

class DiscreteMargins:
    """Get marginal effects of a Discrete Choice model.

    Parameters
    ----------
    results : DiscreteResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """

    def __init__(self, results, args, kwargs={}):
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)

    def summary_frame(self, alpha=0.05):
        """
        Returns a DataFrame summarizing the marginal effects.

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        frame : DataFrames
            A DataFrame summarizing the marginal effects.

        Notes
        -----
        The dataframe is created on each call and not cached, as are the
        tables build in `summary()`
        """
        pass

    def conf_int(self, alpha=0.05):
        """
        Returns the confidence intervals of the marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        conf_int : ndarray
            An array with lower, upper confidence intervals for the marginal
            effects.
        """
        pass

    def summary(self, alpha=0.05):
        """
        Returns a summary table for marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        Summary : SummaryTable
            A SummaryTable instance
        """
        pass

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        pass