"""
Tools for working with dates
"""
from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime
_quarter_to_day = {'1': (3, 31), '2': (6, 30), '3': (9, 30), '4': (12, 31), 'I': (3, 31), 'II': (6, 30), 'III': (9, 30), 'IV': (12, 31)}
_mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_months_with_days = lzip(lrange(1, 13), _mdays)
_month_to_day = dict(zip(map(str, lrange(1, 13)), _months_with_days))
_month_to_day.update(dict(zip(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'], _months_with_days)))
_y_pattern = '^\\d?\\d?\\d?\\d$'
_q_pattern = '\n^               # beginning of string\n\\d?\\d?\\d?\\d     # match any number 1-9999, includes leading zeros\n\n(:?q)           # use q or a : as a separator\n\n([1-4]|(I{1,3}V?)) # match 1-4 or I-IV roman numerals\n\n$               # end of string\n'
_m_pattern = '\n^               # beginning of string\n\\d?\\d?\\d?\\d     # match any number 1-9999, includes leading zeros\n\n(:?m)           # use m or a : as a separator\n\n(([1-9][0-2]?)|(I?XI{0,2}|I?VI{0,3}|I{1,3}))  # match 1-12 or\n                                              # I-XII roman numerals\n\n$               # end of string\n'

def date_parser(timestr, parserinfo=None, **kwargs):
    """
    Uses dateutil.parser.parse, but also handles monthly dates of the form
    1999m4, 1999:m4, 1999:mIV, 1999mIV and the same for quarterly data
    with q instead of m. It is not case sensitive. The default for annual
    data is the end of the year, which also differs from dateutil.
    """
    pass

def date_range_str(start, end=None, length=None):
    """
    Returns a list of abbreviated date strings.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Returns
    -------
    date_range : list
        List of strings
    """
    pass

def dates_from_str(dates):
    """
    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    dates : array_like
        A sequence of abbreviated dates as string. For instance,
        '1996m1' or '1996Q1'. The datetime dates are at the end of the
        period.

    Returns
    -------
    date_list : ndarray
        A list of datetime types.
    """
    pass

def dates_from_range(start, end=None, length=None):
    """
    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> nobs = 50
    >>> dates = pd.date_range('1960m1', length=nobs)


    Returns
    -------
    date_list : ndarray
        A list of datetime types.
    """
    pass