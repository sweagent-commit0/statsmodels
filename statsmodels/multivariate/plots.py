import matplotlib.pyplot as plt
import numpy as np

def plot_scree(eigenvals, total_var, ncomp=None, x_label='factor'):
    """
    Plot of the ordered eigenvalues and variance explained for the loadings

    Parameters
    ----------
    eigenvals : array_like
        The eigenvalues
    total_var : float
        the total variance (for plotting percent variance explained)
    ncomp : int, optional
        Number of factors to include in the plot.  If None, will
        included the same as the number of maximum possible loadings
    x_label : str
        label of x-axis

    Returns
    -------
    Figure
        Handle to the figure.
    """
    pass

def plot_loadings(loadings, col_names=None, row_names=None, loading_pairs=None, percent_variance=None, title='Factor patterns'):
    """
    Plot factor loadings in 2-d plots

    Parameters
    ----------
    loadings : array like
        Each column is a component (or factor)
    col_names : a list of strings
        column names of `loadings`
    row_names : a list of strings
        row names of `loadings`
    loading_pairs : None or a list of tuples
        Specify plots. Each tuple (i, j) represent one figure, i and j is
        the loading number for x-axis and y-axis, respectively. If `None`,
        all combinations of the loadings will be plotted.
    percent_variance : array_like
        The percent variance explained by each factor.

    Returns
    -------
    figs : a list of figure handles
    """
    pass