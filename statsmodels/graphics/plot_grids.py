"""create scatterplot with confidence ellipsis

Author: Josef Perktold
License: BSD-3

TODO: update script to use sharex, sharey, and visible=False
    see https://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label
    for sharex I need to have the ax of the last_row when editing the earlier
    rows. Or you axes_grid1, imagegrid
    http://matplotlib.sourceforge.net/mpl_toolkits/axes_grid/users/overview.html
"""
import numpy as np
from scipy import stats
from . import utils
__all__ = ['scatter_ellipse']

def _make_ellipse(mean, cov, ax, level=0.95, color=None):
    """Support function for scatter_ellipse."""
    pass

def scatter_ellipse(data, level=0.9, varnames=None, ell_kwds=None, plot_kwds=None, add_titles=False, keep_ticks=False, fig=None):
    """Create a grid of scatter plots with confidence ellipses.

    ell_kwds, plot_kdes not used yet

    looks ok with 5 or 6 variables, too crowded with 8, too empty with 1

    Parameters
    ----------
    data : array_like
        Input data.
    level : scalar, optional
        Default is 0.9.
    varnames : list[str], optional
        Variable names.  Used for y-axis labels, and if `add_titles` is True
        also for titles.  If not given, integers 1..data.shape[1] are used.
    ell_kwds : dict, optional
        UNUSED
    plot_kwds : dict, optional
        UNUSED
    add_titles : bool, optional
        Whether or not to add titles to each subplot.  Default is False.
        Titles are constructed from `varnames`.
    keep_ticks : bool, optional
        If False (default), remove all axis ticks.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise `fig` itself.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> from statsmodels.graphics.plot_grids import scatter_ellipse
    >>> data = sm.datasets.statecrime.load_pandas().data
    >>> fig = plt.figure(figsize=(8,8))
    >>> scatter_ellipse(data, varnames=data.columns, fig=fig)
    >>> plt.show()

    .. plot:: plots/graphics_plot_grids_scatter_ellipse.py
    """
    pass