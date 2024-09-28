"""
A predict-like function that constructs means and pointwise or
simultaneous confidence bands for the function f(x) = E[Y | X*=x,
X1=x1, ...], where X* is the focus variable and X1, X2, ... are
non-focus variables.  This is especially useful when conducting a
functional regression in which the role of x is modeled with b-splines
or other basis functions.
"""
import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
_predict_functional_doc = "\n    Predictions and contrasts of a fitted model as a function of a given covariate.\n\n    The value of the focus variable varies along a sequence of its\n    quantiles, calculated from the data used to fit the model.  The\n    other variables are held constant either at given values, or at\n    values obtained by applying given summary functions to the data\n    used to fit the model.  Optionally, a second specification of the\n    non-focus variables is provided and the contrast between the two\n    specifications is returned.\n\n    Parameters\n    ----------\n    result : statsmodels result object\n        A results object for the fitted model.\n    focus_var : str\n        The name of the 'focus variable'.\n    summaries : dict-like\n        A map from names of non-focus variables to summary functions.\n        Each summary function is applied to the data used to fit the\n        model, to obtain a value at which the variable is held fixed.\n    values : dict-like\n        Values at which a given non-focus variable is held fixed.\n    summaries2 : dict-like\n        A second set of summary functions used to define a contrast.\n    values2 : dict-like\n        A second set of fixed values used to define a contrast.\n    alpha : float\n        `1 - alpha` is the coverage probability.\n    ci_method : str\n        The method for constructing the confidence band, one of\n        'pointwise', 'scheffe', and 'simultaneous'.\n    num_points : int\n        The number of equally-spaced quantile points where the\n        prediction is made.\n    exog : array_like\n        Explicitly provide points to cover with the confidence band.\n    exog2 : array_like\n        Explicitly provide points to contrast to `exog` in a functional\n        confidence band.\n    kwargs :\n        Arguments passed to the `predict` method.\n\n    Returns\n    -------\n    pred : array_like\n        The predicted mean values.\n    cb : array_like\n        An array with two columns, containing respectively the lower\n        and upper limits of a confidence band.\n    fvals : array_like\n        The values of the focus variable at which the prediction is\n        made.\n\n    Notes\n    -----\n    All variables in the model except for the focus variable should be\n    included as a key in either `summaries` or `values` (unless `exog`\n    is provided).\n\n    If `summaries2` and `values2` are not provided, the returned value\n    contains predicted conditional means for the outcome as the focus\n    variable varies, with the other variables fixed as specified.\n\n    If `summaries2` and/or `values2` is provided, two sets of\n    predicted conditional means are calculated, and the returned value\n    is the contrast between them.\n\n    If `exog` is provided, then the rows should contain a sequence of\n    values approximating a continuous path through the domain of the\n    covariates.  For example, if Z(s) is the covariate expressed as a\n    function of s, then the rows of exog may approximate Z(g(s)) for\n    some continuous function g.  If `exog` is provided then neither of\n    the summaries or values arguments should be provided.  If `exog2`\n    is also provided, then the returned value is a contrast between\n    the functionas defined by `exog` and `exog2`.\n\n    Examples\n    --------\n    Fit a model using a formula in which the predictors are age\n    (modeled with splines), ethnicity (which is categorical), gender,\n    and income.  Then we obtain the fitted mean values as a function\n    of age for females with mean income and the most common\n    ethnicity.\n\n    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + C(ethnicity) + gender + income', data)\n    >>> result = model.fit()\n    >>> mode = lambda x : x.value_counts().argmax()\n    >>> summaries = {'income': np.mean, ethnicity=mode}\n    >>> values = {'gender': 'female'}\n    >>> pr, cb, x = predict_functional(result, 'age', summaries, values)\n\n    Fit a model using arrays.  Plot the means as a function of x3,\n    holding x1 fixed at its mean value in the data used to fit the\n    model, and holding x2 fixed at 1.\n\n    >>> model = sm.OLS(y ,x)\n    >>> result = model.fit()\n    >>> summaries = {'x1': np.mean}\n    >>> values = {'x2': 1}\n    >>> pr, cb, x = predict_functional(result, 'x3', summaries, values)\n\n    Fit a model usng a formula and construct a contrast comparing the\n    female and male predicted mean functions.\n\n    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + gender', data)\n    >>> result = model.fit()\n    >>> values = {'gender': 'female'}\n    >>> values2 = {'gender': 'male'}\n    >>> pr, cb, x = predict_functional(result, 'age', values=values, values2=values2)\n    "

def _make_exog_from_formula(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit with a formula.

    Returns
    -------
    dexog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    fexog : data frame
        The data frame `dexog` processed through the model formula.
    """
    pass

def _make_exog_from_arrays(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit without a formula.

    Returns
    -------
    exog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    """
    pass

def _glm_basic_scr(result, exog, alpha):
    """
    The basic SCR from (Sun et al. Annals of Statistics 2000).

    Computes simultaneous confidence regions (SCR).

    Parameters
    ----------
    result : results instance
        The fitted GLM results instance
    exog : array_like
        The exog values spanning the interval
    alpha : float
        `1 - alpha` is the coverage probability.

    Returns
    -------
    An array with two columns, containing the lower and upper
    confidence bounds, respectively.

    Notes
    -----
    The rows of `exog` should be a sequence of covariate values
    obtained by taking one 'free variable' x and varying it over an
    interval.  The matrix `exog` is thus the basis functions and any
    other covariates evaluated as x varies.
    """
    pass