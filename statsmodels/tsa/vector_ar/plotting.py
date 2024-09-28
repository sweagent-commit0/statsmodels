from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util

class MPLConfigurator:

    def __init__(self):
        self._inverse_actions = []

def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series
    """
    pass

def plot_with_error(y, error, x=None, axes=None, value_fmt='k', error_fmt='k--', alpha=0.05, stderr_type='asym'):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None
    """
    pass

def plot_full_acorr(acorr, fontsize=8, linewidth=8, xlabel=None, err_bound=None):
    """

    Parameters
    ----------
    """
    pass

def irf_grid_plot(values, stderr, impcol, rescol, names, title, signif=0.05, hlines=None, subplot_params=None, plot_params=None, figsize=(10, 10), stderr_type='asym'):
    """
    Reusable function to make flexible grid plots of impulse responses and
    comulative effects

    values : (T + 1) x k x k
    stderr : T x k x k
    hlines : k x k
    """
    pass