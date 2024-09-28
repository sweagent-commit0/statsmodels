from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt

class Summary:

    def __init__(self):
        self.tables = []
        self.settings = []
        self.extra_txt = []
        self.title = None
        self._merge_latex = False

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        pass

    def _repr_latex_(self):
        """Display as LaTeX when converting IPython notebook to PDF."""
        pass

    def add_df(self, df, index=True, header=True, float_format='%.4f', align='r'):
        """
        Add the contents of a DataFrame to summary table

        Parameters
        ----------
        df : DataFrame
        header : bool
            Reproduce the DataFrame column labels in summary table
        index : bool
            Reproduce the DataFrame row labels in summary table
        float_format : str
            Formatting to float data columns
        align : str
            Data alignment (l/c/r)
        """
        pass

    def add_array(self, array, align='r', float_format='%.4f'):
        """Add the contents of a Numpy array to summary table

        Parameters
        ----------
        array : numpy array (2D)
        float_format : str
            Formatting to array if type is float
        align : str
            Data alignment (l/c/r)
        """
        pass

    def add_dict(self, d, ncols=2, align='l', float_format='%.4f'):
        """Add the contents of a Dict to summary table

        Parameters
        ----------
        d : dict
            Keys and values are automatically coerced to strings with str().
            Users are encouraged to format them before using add_dict.
        ncols : int
            Number of columns of the output table
        align : str
            Data alignment (l/c/r)
        float_format : str
            Formatting to float data columns
        """
        pass

    def add_text(self, string):
        """Append a note to the bottom of the summary table. In ASCII tables,
        the note will be wrapped to table width. Notes are not indented.
        """
        pass

    def add_title(self, title=None, results=None):
        """Insert a title on top of the summary table. If a string is provided
        in the title argument, that string is printed. If no title string is
        provided but a results instance is provided, statsmodels attempts
        to construct a useful title automatically.
        """
        pass

    def add_base(self, results, alpha=0.05, float_format='%.4f', title=None, xname=None, yname=None):
        """Try to construct a basic summary instance.

        Parameters
        ----------
        results : Model results instance
        alpha : float
            significance level for the confidence intervals (optional)
        float_format: str
            Float formatting for summary of parameters (optional)
        title : str
            Title of the summary table (optional)
        xname : list[str] of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : str
            Name of the dependent variable (optional)
        """
        pass

    def as_text(self):
        """Generate ASCII Summary Table
        """
        pass

    def as_html(self):
        """Generate HTML Summary Table
        """
        pass

    def as_latex(self, label=''):
        """Generate LaTeX Summary Table

        Parameters
        ----------
        label : str
            Label of the summary table that can be referenced
            in a latex document (optional)
        """
        pass

def _measure_tables(tables, settings):
    """Compare width of ascii tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    """
    pass
_model_types = {'OLS': 'Ordinary least squares', 'GLS': 'Generalized least squares', 'GLSAR': 'Generalized least squares with AR(p)', 'WLS': 'Weighted least squares', 'RLM': 'Robust linear model', 'NBin': 'Negative binomial model', 'GLM': 'Generalized linear model'}

def summary_model(results):
    """
    Create a dict with information about the model
    """
    pass

def summary_params(results, yname=None, xname=None, alpha=0.05, use_t=True, skip_header=False, float_format='%.4f'):
    """create a summary table of parameters from results instance

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : {str, None}
        optional name for the endogenous variable, default is "y"
    xname : {list[str], None}
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_header : bool
        If false (default), then the header row is added. If true, then no
        header row is added.
    float_format : str
        float formatting options (e.g. ".3g")

    Returns
    -------
    params_table : SimpleTable instance
    """
    pass

def _col_params(result, float_format='%.4f', stars=True, include_r2=False):
    """Stack coefficients and standard errors in single column
    """
    pass

def _col_info(result, info_dict=None):
    """Stack model info in a column
    """
    pass

def summary_col(results, float_format='%.4f', model_names=(), stars=False, info_dict=None, regressor_order=(), drop_omitted=False, include_r2=True):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : str, optional
        float format for coefficients and standard errors
        Default : '%.4f'
    model_names : list[str], optional
        Must have same length as the number of results. If the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict, default None
        dict of functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":lambda x:(x.nobs), "R2": ..., "OLS":{
        "R2":...}}` would only show `R2` for OLS regression models, but
        additionally `N` for all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list[str], optional
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    drop_omitted : bool, optional
        Includes regressors that are not specified in regressor_order. If
        False, regressors not specified will be appended to end of the list.
        If True, only regressors in regressor_order will be included.
    include_r2 : bool, optional
        Includes R2 and adjusted R2 in the summary table.
    """
    pass