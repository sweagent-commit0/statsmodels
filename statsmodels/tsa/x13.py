"""
Run x12/x13-arima specs in a subprocess from Python and curry results back
into python.

Notes
-----
Many of the functions are called x12. However, they are also intended to work
for x13. If this is not the case, it's a bug.
"""
from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import X13NotFoundError, IOWarning, X13Error, X13Warning
__all__ = ['x13_arima_select_order', 'x13_arima_analysis']
_binary_names = ('x13as.exe', 'x13as', 'x12a.exe', 'x12a', 'x13as_ascii', 'x13as_html')

class _freq_to_period:

    def __getitem__(self, key):
        if key.startswith('M'):
            return 12
        elif key.startswith('Q'):
            return 4
        elif key.startswith('W'):
            return 52
_freq_to_period = _freq_to_period()
_period_to_freq = {12: 'M', 4: 'Q'}
_log_to_x12 = {True: 'log', False: 'none', None: 'auto'}
_bool_to_yes_no = lambda x: 'yes' if x else 'no'

def _find_x12(x12path=None, prefer_x13=True):
    """
    If x12path is not given, then either x13as[.exe] or x12a[.exe] must
    be found on the PATH. Otherwise, the environmental variable X12PATH or
    X13PATH must be defined. If prefer_x13 is True, only X13PATH is searched
    for. If it is false, only X12PATH is searched for.
    """
    pass

def _clean_order(order):
    """
    Takes something like (1 1 0)(0 1 1) and returns a arma order, sarma
    order tuple. Also accepts (1 1 0) and return arma order and (0, 0, 0)
    """
    pass

def _convert_out_to_series(x, dates, name):
    """
    Convert x to a DataFrame where x is a string in the format given by
    x-13arima-seats output.
    """
    pass

class Spec:
    pass

class SeriesSpec(Spec):
    """
    Parameters
    ----------
    data
    appendbcst : bool
    appendfcst : bool
    comptype
    compwt
    decimals
    modelspan
    name
    period
    precision
    to_print
    to_save
    span
    start
    title
    type

    Notes
    -----
    Rarely used arguments

    divpower
    missingcode
    missingval
    saveprecision
    trimzero
    """

    def __init__(self, data, name='Unnamed Series', appendbcst=False, appendfcst=False, comptype=None, compwt=1, decimals=0, modelspan=(), period=12, precision=0, to_print=[], to_save=[], span=(), start=(1, 1), title='', series_type=None, divpower=None, missingcode=-99999, missingval=1000000000):
        appendbcst, appendfcst = map(_bool_to_yes_no, [appendbcst, appendfcst])
        series_name = '"{0}"'.format(name[:64])
        title = '"{0}"'.format(title[:79])
        self.set_options(data=data, appendbcst=appendbcst, appendfcst=appendfcst, period=period, start=start, title=title, name=series_name)

@deprecate_kwarg('forecast_years', 'forecast_periods')
def x13_arima_analysis(endog, maxorder=(2, 1), maxdiff=(2, 1), diff=None, exog=None, log=None, outlier=True, trading=False, forecast_periods=None, retspec=False, speconly=False, start=None, freq=None, print_stdout=False, x12path=None, prefer_x13=True, tempdir=None):
    """
    Perform x13-arima analysis for monthly or quarterly data.

    Parameters
    ----------
    endog : array_like, pandas.Series
        The series to model. It is best to use a pandas object with a
        DatetimeIndex or PeriodIndex. However, you can pass an array-like
        object. If your object does not have a dates index then ``start`` and
        ``freq`` are not optional.
    maxorder : tuple
        The maximum order of the regular and seasonal ARMA polynomials to
        examine during the model identification. The order for the regular
        polynomial must be greater than zero and no larger than 4. The
        order for the seasonal polynomial may be 1 or 2.
    maxdiff : tuple
        The maximum orders for regular and seasonal differencing in the
        automatic differencing procedure. Acceptable inputs for regular
        differencing are 1 and 2. The maximum order for seasonal differencing
        is 1. If ``diff`` is specified then ``maxdiff`` should be None.
        Otherwise, ``diff`` will be ignored. See also ``diff``.
    diff : tuple
        Fixes the orders of differencing for the regular and seasonal
        differencing. Regular differencing may be 0, 1, or 2. Seasonal
        differencing may be 0 or 1. ``maxdiff`` must be None, otherwise
        ``diff`` is ignored.
    exog : array_like
        Exogenous variables.
    log : bool or None
        If None, it is automatically determined whether to log the series or
        not. If False, logs are not taken. If True, logs are taken.
    outlier : bool
        Whether or not outliers are tested for and corrected, if detected.
    trading : bool
        Whether or not trading day effects are tested for.
    forecast_periods : int
        Number of forecasts produced. The default is None.
    retspec : bool
        Whether to return the created specification file. Can be useful for
        debugging.
    speconly : bool
        Whether to create the specification file and then return it without
        performing the analysis. Can be useful for debugging.
    start : str, datetime
        Must be given if ``endog`` does not have date information in its index.
        Anything accepted by pandas.DatetimeIndex for the start value.
    freq : str
        Must be givein if ``endog`` does not have date information in its
        index. Anything accepted by pandas.DatetimeIndex for the freq value.
    print_stdout : bool
        The stdout from X12/X13 is suppressed. To print it out, set this
        to True. Default is False.
    x12path : str or None
        The path to x12 or x13 binary. If None, the program will attempt
        to find x13as or x12a on the PATH or by looking at X13PATH or
        X12PATH depending on the value of prefer_x13.
    prefer_x13 : bool
        If True, will look for x13as first and will fallback to the X13PATH
        environmental variable. If False, will look for x12a first and will
        fallback to the X12PATH environmental variable. If x12path points
        to the path for the X12/X13 binary, it does nothing.
    tempdir : str
        The path to where temporary files are created by the function.
        If None, files are created in the default temporary file location.

    Returns
    -------
    Bunch
        A bunch object containing the listed attributes.

        - results : str
          The full output from the X12/X13 run.
        - seasadj : pandas.Series
          The final seasonally adjusted ``endog``.
        - trend : pandas.Series
          The trend-cycle component of ``endog``.
        - irregular : pandas.Series
          The final irregular component of ``endog``.
        - stdout : str
          The captured stdout produced by x12/x13.
        - spec : str, optional
          Returned if ``retspec`` is True. The only thing returned if
          ``speconly`` is True.

    Notes
    -----
    This works by creating a specification file, writing it to a temporary
    directory, invoking X12/X13 in a subprocess, and reading the output
    directory, invoking exog12/X13 in a subprocess, and reading the output
    back in.
    """
    pass

@deprecate_kwarg('forecast_years', 'forecast_periods')
def x13_arima_select_order(endog, maxorder=(2, 1), maxdiff=(2, 1), diff=None, exog=None, log=None, outlier=True, trading=False, forecast_periods=None, start=None, freq=None, print_stdout=False, x12path=None, prefer_x13=True, tempdir=None):
    """
    Perform automatic seasonal ARIMA order identification using x12/x13 ARIMA.

    Parameters
    ----------
    endog : array_like, pandas.Series
        The series to model. It is best to use a pandas object with a
        DatetimeIndex or PeriodIndex. However, you can pass an array-like
        object. If your object does not have a dates index then ``start`` and
        ``freq`` are not optional.
    maxorder : tuple
        The maximum order of the regular and seasonal ARMA polynomials to
        examine during the model identification. The order for the regular
        polynomial must be greater than zero and no larger than 4. The
        order for the seasonal polynomial may be 1 or 2.
    maxdiff : tuple
        The maximum orders for regular and seasonal differencing in the
        automatic differencing procedure. Acceptable inputs for regular
        differencing are 1 and 2. The maximum order for seasonal differencing
        is 1. If ``diff`` is specified then ``maxdiff`` should be None.
        Otherwise, ``diff`` will be ignored. See also ``diff``.
    diff : tuple
        Fixes the orders of differencing for the regular and seasonal
        differencing. Regular differencing may be 0, 1, or 2. Seasonal
        differencing may be 0 or 1. ``maxdiff`` must be None, otherwise
        ``diff`` is ignored.
    exog : array_like
        Exogenous variables.
    log : bool or None
        If None, it is automatically determined whether to log the series or
        not. If False, logs are not taken. If True, logs are taken.
    outlier : bool
        Whether or not outliers are tested for and corrected, if detected.
    trading : bool
        Whether or not trading day effects are tested for.
    forecast_periods : int
        Number of forecasts produced. The default is None.
    start : str, datetime
        Must be given if ``endog`` does not have date information in its index.
        Anything accepted by pandas.DatetimeIndex for the start value.
    freq : str
        Must be givein if ``endog`` does not have date information in its
        index. Anything accepted by pandas.DatetimeIndex for the freq value.
    print_stdout : bool
        The stdout from X12/X13 is suppressed. To print it out, set this
        to True. Default is False.
    x12path : str or None
        The path to x12 or x13 binary. If None, the program will attempt
        to find x13as or x12a on the PATH or by looking at X13PATH or X12PATH
        depending on the value of prefer_x13.
    prefer_x13 : bool
        If True, will look for x13as first and will fallback to the X13PATH
        environmental variable. If False, will look for x12a first and will
        fallback to the X12PATH environmental variable. If x12path points
        to the path for the X12/X13 binary, it does nothing.
    tempdir : str
        The path to where temporary files are created by the function.
        If None, files are created in the default temporary file location.

    Returns
    -------
    Bunch
        A bunch object containing the listed attributes.

        - order : tuple
          The regular order.
        - sorder : tuple
          The seasonal order.
        - include_mean : bool
          Whether to include a mean or not.
        - results : str
          The full results from the X12/X13 analysis.
        - stdout : str
          The captured stdout from the X12/X13 analysis.

    Notes
    -----
    This works by creating a specification file, writing it to a temporary
    directory, invoking X12/X13 in a subprocess, and reading the output back
    in.
    """
    pass

class X13ArimaAnalysisResult:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)