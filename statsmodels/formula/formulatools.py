import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
formula_handler = {}

class NAAction(NAAction):
    pass

def handle_formula_data(Y, X, formula, depth=0, missing='drop'):
    """
    Returns endog, exog, and the model specification from arrays and formula.

    Parameters
    ----------
    Y : array_like
        Either endog (the LHS) of a model specification or all of the data.
        Y must define __getitem__ for now.
    X : array_like
        Either exog or None. If all the data for the formula is provided in
        Y then you must explicitly set X to None.
    formula : str or patsy.model_desc
        You can pass a handler by import formula_handler and adding a
        key-value pair where the key is the formula object class and
        the value is a function that returns endog, exog, formula object.

    Returns
    -------
    endog : array_like
        Should preserve the input type of Y,X.
    exog : array_like
        Should preserve the input type of Y,X. Could be None.
    """
    pass

def _remove_intercept_patsy(terms):
    """
    Remove intercept from Patsy terms.
    """
    pass

def _intercept_idx(design_info):
    """
    Returns boolean array index indicating which column holds the intercept.
    """
    pass

def make_hypotheses_matrices(model_results, test_formula):
    """
    """
    pass