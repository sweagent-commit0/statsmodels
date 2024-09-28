from functools import wraps
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period

def pandas_wrapper_freq(func, trim_head=None, trim_tail=None, freq_kw='freq', columns=None, *args, **kwargs):
    """
    Return a new function that catches the incoming X, checks if it's pandas,
    calls the functions as is. Then wraps the results in the incoming index.

    Deals with frequencies. Expects that the function returns a tuple,
    a Bunch object, or a pandas-object.
    """
    pass