from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import _has_intercept, _intercept_idx, _remove_intercept_patsy
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS

def anova_single(model, **kwargs):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model
    typ : int or str {1,2,3} or {"I","II","III"}
        Type of sum of squares to use.

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    pass

def anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test, pr_test, robust):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    pass

def anova2_lm_single(model, design_info, n_rows, test, pr_test, robust):
    """
    Anova type II table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.

    Type II
    Sum of Squares compares marginal contribution of terms. Thus, it is
    not particularly useful for models with significant interaction terms.
    """
    pass

def anova_lm(*args, **kwargs):
    """
    Anova table for one or more fitted linear models.

    Parameters
    ----------
    args : fitted linear model results instance
        One or more fitted linear models
    scale : float
        Estimate of variance, If None, will be estimated from the largest
        model. Default is None.
    test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".
    typ : str or int {"I","II","III"} or {1,2,3}
        The type of Anova test to perform. See notes.
    robust : {None, "hc0", "hc1", "hc2", "hc3"}
        Use heteroscedasticity-corrected coefficient covariance matrix.
        If robust covariance is desired, it is recommended to use `hc3`.

    Returns
    -------
    anova : DataFrame
        When args is a single model, return is DataFrame with columns:

        sum_sq : float64
            Sum of squares for model terms.
        df : float64
            Degrees of freedom for model terms.
        F : float64
            F statistic value for significance of adding model terms.
        PR(>F) : float64
            P-value for significance of adding model terms.

        When args is multiple models, return is DataFrame with columns:

        df_resid : float64
            Degrees of freedom of residuals in models.
        ssr : float64
            Sum of squares of residuals in models.
        df_diff : float64
            Degrees of freedom difference from previous model in args
        ss_dff : float64
            Difference in ssr from previous model in args
        F : float64
            F statistic comparing to previous model in args
        PR(>F): float64
            P-value for significance comparing to previous model in args

    Notes
    -----
    Model statistics are given in the order of args. Models must have been fit
    using the formula api.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.formula.api import ols
    >>> moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
    >>> data = moore.data
    >>> data = data.rename(columns={"partner.status" :
    ...                             "partner_status"}) # make name pythonic
    >>> moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
    ...                 data=data).fit()
    >>> table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
    >>> print(table)
    """
    pass

def _ssr_reduced_model(y, x, term_slices, params, keys):
    """
    Residual sum of squares of OLS model excluding factors in `keys`
    Assumes x matrix is orthogonal

    Parameters
    ----------
    y : array_like
        dependent variable
    x : array_like
        independent variables
    term_slices : a dict of slices
        term_slices[key] is a boolean array specifies the parameters
        associated with the factor `key`
    params : ndarray
        OLS solution of y = x * params
    keys : keys for term_slices
        factors to be excluded

    Returns
    -------
    rss : float
        residual sum of squares
    df : int
        degrees of freedom
    """
    pass

class AnovaRM:
    """
    Repeated measures Anova using least squares regression

    The full model regression residual sum of squares is
    used to compare with the reduced model for calculating the
    within-subject effect sum of squares [1].

    Currently, only fully balanced within-subject designs are supported.
    Calculation of between-subject effects and corrections for violation of
    sphericity are not yet implemented.

    Parameters
    ----------
    data : DataFrame
    depvar : str
        The dependent variable in `data`
    subject : str
        Specify the subject id
    within : list[str]
        The within-subject factors
    between : list[str]
        The between-subject factors, this is not yet implemented
    aggregate_func : {None, 'mean', callable}
        If the data set contains more than a single observation per subject
        and cell of the specified model, this function will be used to
        aggregate the data before running the Anova. `None` (the default) will
        not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
        An exception will be raised if aggregation is required, but no
        aggregation function was specified.

    Returns
    -------
    results : AnovaResults instance

    Raises
    ------
    ValueError
        If the data need to be aggregated, but `aggregate_func` was not
        specified.

    Notes
    -----
    This implementation currently only supports fully balanced designs. If the
    data contain more than one observation per subject and cell of the design,
    these observations need to be aggregated into a single observation
    before the Anova is calculated, either manually or by passing an aggregation
    function via the `aggregate_func` keyword argument.
    Note that if the input data set was not balanced before performing the
    aggregation, the implied heteroscedasticity of the data is ignored.

    References
    ----------
    .. [*] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.
    """

    def __init__(self, data, depvar, subject, within=None, between=None, aggregate_func=None):
        self.data = data
        self.depvar = depvar
        self.within = within
        if 'C' in within:
            raise ValueError("Factor name cannot be 'C'! This is in conflict with patsy's contrast function name.")
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not yet supported!')
        self.subject = subject
        if aggregate_func == 'mean':
            self.aggregate_func = pd.Series.mean
        else:
            self.aggregate_func = aggregate_func
        if not data.equals(data.drop_duplicates(subset=[subject] + within)):
            if self.aggregate_func is not None:
                self._aggregate()
            else:
                msg = 'The data set contains more than one observation per subject and cell. Either aggregate the data manually, or pass the `aggregate_func` parameter.'
                raise ValueError(msg)
        self._check_data_balanced()

    def _check_data_balanced(self):
        """raise if data is not balanced

        This raises a ValueError if the data is not balanced, and
        returns None if it is balance

        Return might change
        """
        pass

    def fit(self):
        """estimate the model and compute the Anova table

        Returns
        -------
        AnovaResults instance
        """
        pass

class AnovaResults:
    """
    Anova results class

    Attributes
    ----------
    anova_table : DataFrame
    """

    def __init__(self, anova_table):
        self.anova_table = anova_table

    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        """create summary results

        Returns
        -------
        summary : summary2.Summary instance
        """
        pass
if __name__ == '__main__':
    import pandas
    from statsmodels.formula.api import ols
    moore = pandas.read_csv('moore.csv', skiprows=1, names=['partner_status', 'conformity', 'fcategory', 'fscore'])
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)', data=moore).fit()
    mooreB = ols('conformity ~ C(partner_status, Sum)', data=moore).fit()
    table = anova_lm(moore_lm, typ=2)