"""Helper functions for graphics with Matplotlib."""
from statsmodels.compat.python import lrange
__all__ = ['create_mpl_ax', 'create_mpl_fig']

def _import_mpl():
    """This function is not needed outside this utils module."""
    pass

def create_mpl_ax(ax=None):
    """Helper function for when a single plot axis is needed.

    Parameters
    ----------
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    ax : AxesSubplot
        The created axis if `ax` is None, otherwise the axis that was passed
        in.

    Notes
    -----
    This function imports `matplotlib.pyplot`, which should only be done to
    create (a) figure(s) with ``plt.figure``.  All other functionality exposed
    by the pyplot module can and should be imported directly from its
    Matplotlib module.

    See Also
    --------
    create_mpl_fig

    Examples
    --------
    A plotting function has a keyword ``ax=None``.  Then calls:

    >>> from statsmodels.graphics import utils
    >>> fig, ax = utils.create_mpl_ax(ax)
    """
    pass

def create_mpl_fig(fig=None, figsize=None):
    """Helper function for when multiple plot axes are needed.

    Those axes should be created in the functions they are used in, with
    ``fig.add_subplot()``.

    Parameters
    ----------
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise the input `fig` is
        returned.

    See Also
    --------
    create_mpl_ax
    """
    pass

def maybe_name_or_idx(idx, model):
    """
    Give a name or an integer and return the name and integer location of the
    column in a design matrix.
    """
    pass

def get_data_names(series_or_dataframe):
    """
    Input can be an array or pandas-like. Will handle 1d array-like but not
    2d. Returns a str for 1d data or a list of strings for 2d data.
    """
    pass

def annotate_axes(index, labels, points, offset_points, size, ax, **kwargs):
    """
    Annotate Axes with labels, points, offset_points according to the
    given index.
    """
    pass