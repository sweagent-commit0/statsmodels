"""General linear model

author: Yichuan Liu
"""
import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
__docformat__ = 'restructuredtext en'
_hypotheses_doc = 'hypotheses : list[tuple]\n    Hypothesis `L*B*M = C` to be tested where B is the parameters in\n    regression Y = X*B. Each element is a tuple of length 2, 3, or 4:\n\n      * (name, contrast_L)\n      * (name, contrast_L, transform_M)\n      * (name, contrast_L, transform_M, constant_C)\n\n    containing a string `name`, the contrast matrix L, the transform\n    matrix M (for transforming dependent variables), and right-hand side\n    constant matrix constant_C, respectively.\n\n    contrast_L : 2D array or an array of strings\n        Left-hand side contrast matrix for hypotheses testing.\n        If 2D array, each row is an hypotheses and each column is an\n        independent variable. At least 1 row\n        (1 by k_exog, the number of independent variables) is required.\n        If an array of strings, it will be passed to\n        patsy.DesignInfo().linear_constraint.\n\n    transform_M : 2D array or an array of strings or None, optional\n        Left hand side transform matrix.\n        If `None` or left out, it is set to a k_endog by k_endog\n        identity matrix (i.e. do not transform y matrix).\n        If an array of strings, it will be passed to\n        patsy.DesignInfo().linear_constraint.\n\n    constant_C : 2D array or None, optional\n        Right-hand side constant matrix.\n        if `None` or left out it is set to a matrix of zeros\n        Must has the same number of rows as contrast_L and the same\n        number of columns as transform_M\n\n    If `hypotheses` is None: 1) the effect of each independent variable\n    on the dependent variables will be tested. Or 2) if model is created\n    using a formula,  `hypotheses` will be created according to\n    `design_info`. 1) and 2) is equivalent if no additional variables\n    are created by the formula (e.g. dummy variables for categorical\n    variables and interaction terms)\n'

def _multivariate_ols_fit(endog, exog, method='svd', tolerance=1e-08):
    """
    Solve multivariate linear model y = x * params
    where y is dependent variables, x is independent variables

    Parameters
    ----------
    endog : array_like
        each column is a dependent variable
    exog : array_like
        each column is a independent variable
    method : str
        'svd' - Singular value decomposition
        'pinv' - Moore-Penrose pseudoinverse
    tolerance : float, a small positive number
        Tolerance for eigenvalue. Values smaller than tolerance is considered
        zero.
    Returns
    -------
    a tuple of matrices or values necessary for hypotheses testing

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    Notes
    -----
    Status: experimental and incomplete
    """
    pass

def multivariate_stats(eigenvals, r_err_sscp, r_contrast, df_resid, tolerance=1e-08):
    """
    For multivariate linear model Y = X * B
    Testing hypotheses
        L*B*M = 0
    where L is contrast matrix, B is the parameters of the
    multivariate linear model and M is dependent variable transform matrix.
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M

    Parameters
    ----------
    eigenvals : ndarray
        The eigenvalues of inv(E + H)*H
    r_err_sscp : int
        Rank of E + H
    r_contrast : int
        Rank of T matrix
    df_resid : int
        Residual degree of freedom (n_samples minus n_variables of X)
    tolerance : float
        smaller than which eigenvalue is considered 0

    Returns
    -------
    A DataFrame

    References
    ----------
    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    """
    pass

@Substitution(hypotheses_doc=_hypotheses_doc)
def _multivariate_test(hypotheses, exog_names, endog_names, fn):
    """
    Multivariate linear model hypotheses testing

    For y = x * params, where y are the dependent variables and x are the
    independent variables, testing L * params * M = 0 where L is the contrast
    matrix for hypotheses testing and M is the transformation matrix for
    transforming the dependent variables in y.

    Algorithm:
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M
    where H and E correspond to the numerator and denominator of a univariate
    F-test. Then find the eigenvalues of inv(H + E)*H from which the
    multivariate test statistics are calculated.

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML
           /default/viewer.htm#statug_introreg_sect012.htm

    Parameters
    ----------
    %(hypotheses_doc)s
    k_xvar : int
        The number of independent variables
    k_yvar : int
        The number of dependent variables
    fn : function
        a function fn(contrast_L, transform_M) that returns E, H, q, df_resid
        where q is the rank of T matrix

    Returns
    -------
    results : MANOVAResults
    """
    pass

class _MultivariateOLS(Model):
    """
    Multivariate linear model via least squares


    Parameters
    ----------
    endog : array_like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
        variables
    exog : array_like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user (models specified using a formula include an intercept by
        default)

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    """
    _formula_max_endog = None

    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if len(endog.shape) == 1 or endog.shape[1] == 1:
            raise ValueError('There must be more than one dependent variable to fit multivariate OLS!')
        super(_MultivariateOLS, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)

class _MultivariateOLSResults:
    """
    _MultivariateOLS results class
    """

    def __init__(self, fitted_mv_ols):
        if hasattr(fitted_mv_ols, 'data') and hasattr(fitted_mv_ols.data, 'design_info'):
            self.design_info = fitted_mv_ols.data.design_info
        else:
            self.design_info = None
        self.exog_names = fitted_mv_ols.exog_names
        self.endog_names = fitted_mv_ols.endog_names
        self._fittedmod = fitted_mv_ols._fittedmod

    def __str__(self):
        return self.summary().__str__()

    @Substitution(hypotheses_doc=_hypotheses_doc)
    def mv_test(self, hypotheses=None, skip_intercept_test=False):
        """
        Linear hypotheses testing

        Parameters
        ----------
        %(hypotheses_doc)s
        skip_intercept_test : bool
            If true, then testing the intercept is skipped, the model is not
            changed.
            Note: If a term has a numerically insignificant effect, then
            an exception because of emtpy arrays may be raised. This can
            happen for the intercept if the data has been demeaned.

        Returns
        -------
        results: _MultivariateOLSResults

        Notes
        -----
        Tests hypotheses of the form

            L * params * M = C

        where `params` is the regression coefficient matrix for the
        linear model y = x * params, `L` is the contrast matrix, `M` is the
        dependent variable transform matrix and C is the constant matrix.
        """
        pass

class MultivariateTestResults:
    """
    Multivariate test results class

    Returned by `mv_test` method of `_MultivariateOLSResults` class

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary containing test results. See the description
        below for the expected format.
    endog_names : sequence[str]
        A list or other sequence of endogenous variables names
    exog_names : sequence[str]
        A list of other sequence of exogenous variables names

    Attributes
    ----------
    results : dict
        Each hypothesis is contained in a single`key`. Each test must
        have the following keys:

        * 'stat' - contains the multivariate test results
        * 'contrast_L' - contains the contrast_L matrix
        * 'transform_M' - contains the transform_M matrix
        * 'constant_C' - contains the constant_C matrix
        * 'H' - contains an intermediate Hypothesis matrix,
          or the between groups sums of squares and cross-products matrix,
          corresponding to the numerator of the univariate F test.
        * 'E' - contains an intermediate Error matrix,
          corresponding to the denominator of the univariate F test.
          The Hypotheses and Error matrices can be used to calculate
          the same test statistics in 'stat', as well as to calculate
          the discriminant function (canonical correlates) from the
          eigenvectors of inv(E)H.

    endog_names : list[str]
        The endogenous names
    exog_names : list[str]
        The exogenous names
    summary_frame : DataFrame
        Returns results as a MultiIndex DataFrame
    """

    def __init__(self, results, endog_names, exog_names):
        self.results = results
        self.endog_names = list(endog_names)
        self.exog_names = list(exog_names)

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    @property
    def summary_frame(self):
        """
        Return results as a multiindex dataframe
        """
        pass

    def summary(self, show_contrast_L=False, show_transform_M=False, show_constant_C=False):
        """
        Summary of test results

        Parameters
        ----------
        show_contrast_L : bool
            Whether to show contrast_L matrix
        show_transform_M : bool
            Whether to show transform_M matrix
        show_constant_C : bool
            Whether to show the constant_C
        """
        pass