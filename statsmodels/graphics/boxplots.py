"""Variations on boxplots."""
import numpy as np
from scipy.stats import gaussian_kde
from . import utils
__all__ = ['violinplot', 'beanplot']

def violinplot(data, ax=None, labels=None, positions=None, side='both', show_boxplot=True, plot_opts=None):
    """
    Make a violin plot of each dataset in the `data` sequence.

    A violin plot is a boxplot combined with a kernel density estimate of the
    probability density function per point.

    Parameters
    ----------
    data : sequence[array_like]
        Data arrays, one array per value in `positions`.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list[str], optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    show_boxplot : bool, optional
        Whether or not to show normal box plots on top of the violins.
        Default is True.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'violin_fc', MPL color.  Fill color for violins.  Default is 'y'.
          - 'violin_ec', MPL color.  Edge color for violins.  Default is 'k'.
          - 'violin_lw', scalar.  Edge linewidth for violins.  Default is 1.
          - 'violin_alpha', float.  Transparancy of violins.  Default is 0.5.
          - 'cutoff', bool.  If True, limit violin range to data range.
                Default is False.
          - 'cutoff_val', scalar.  Where to cut off violins if `cutoff` is
                True.  Default is 1.5 standard deviations.
          - 'cutoff_type', {'std', 'abs'}.  Whether cutoff value is absolute,
                or in standard deviations.  Default is 'std'.
          - 'violin_width' : float.  Relative width of violins.  Max available
                space is 1, default is 0.8.
          - 'label_fontsize', MPL fontsize.  Adjusts fontsize only if given.
          - 'label_rotation', scalar.  Adjusts label rotation only if given.
                Specify in degrees.
          - 'bw_factor', Adjusts the scipy gaussian_kde kernel. default: None.
                Options for scalar or callable.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    beanplot : Bean plot, builds on `violinplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    Notes
    -----
    The appearance of violins can be customized with `plot_opts`.  If
    customization of boxplot elements is required, set `show_boxplot` to False
    and plot it on top of the violins by calling the Matplotlib `boxplot`
    function directly.  For example::

        violinplot(data, ax=ax, show_boxplot=False)
        ax.boxplot(data, sym='cv', whis=2.5)

    It can happen that the axis labels or tick labels fall outside the plot
    area, especially with rotated labels on the horizontal axis.  With
    Matplotlib 1.1 or higher, this can easily be fixed by calling
    ``ax.tight_layout()``.  With older Matplotlib one has to use ``plt.rc`` or
    ``plt.rcParams`` to fix this, for example::

        plt.rc('figure.subplot', bottom=0.25)
        violinplot(data, ax=ax)

    References
    ----------
    J.L. Hintze and R.D. Nelson, "Violin Plots: A Box Plot-Density Trace
    Synergism", The American Statistician, Vol. 52, pp.181-84, 1998.

    Examples
    --------
    We use the American National Election Survey 1996 dataset, which has Party
    Identification of respondents as independent variable and (among other
    data) age as dependent variable.

    >>> data = sm.datasets.anes96.load_pandas()
    >>> party_ID = np.arange(7)
    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
    ...           "Independent-Indpendent", "Independent-Republican",
    ...           "Weak Republican", "Strong Republican"]

    Group age by party ID, and create a violin plot with it:

    >>> plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
    >>> age = [data.exog['age'][data.endog == id] for id in party_ID]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> sm.graphics.violinplot(age, ax=ax, labels=labels,
    ...                        plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
    ...                                   'label_fontsize':'small',
    ...                                   'label_rotation':30})
    >>> ax.set_xlabel("Party identification of respondent.")
    >>> ax.set_ylabel("Age")
    >>> plt.show()

    .. plot:: plots/graphics_boxplot_violinplot.py
    """
    pass

def _single_violin(ax, pos, pos_data, width, side, plot_opts):
    """"""
    pass

def _set_ticks_labels(ax, data, labels, positions, plot_opts):
    """Set ticks and labels on horizontal axis."""
    pass

def beanplot(data, ax=None, labels=None, positions=None, side='both', jitter=False, plot_opts={}):
    """
    Bean plot of each dataset in a sequence.

    A bean plot is a combination of a `violinplot` (kernel density estimate of
    the probability density function per point) with a line-scatter plot of all
    individual data points.

    Parameters
    ----------
    data : sequence[array_like]
        Data arrays, one array per value in `positions`.
    ax : AxesSubplot
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list[str], optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    jitter : bool, optional
        If True, jitter markers within violin instead of plotting regular lines
        around the center.  This can be useful if the data is very dense.
    plot_opts : dict, optional
        A dictionary with plotting options.  All the options for `violinplot`
        can be specified, they will simply be passed to `violinplot`.  Options
        specific to `beanplot` are:

          - 'violin_width' : float.  Relative width of violins.  Max available
                space is 1, default is 0.8.
          - 'bean_color', MPL color.  Color of bean plot lines.  Default is 'k'.
                Also used for jitter marker edge color if `jitter` is True.
          - 'bean_size', scalar.  Line length as a fraction of maximum length.
                Default is 0.5.
          - 'bean_lw', scalar.  Linewidth, default is 0.5.
          - 'bean_show_mean', bool.  If True (default), show mean as a line.
          - 'bean_show_median', bool.  If True (default), show median as a
                marker.
          - 'bean_mean_color', MPL color.  Color of mean line.  Default is 'b'.
          - 'bean_mean_lw', scalar.  Linewidth of mean line, default is 2.
          - 'bean_mean_size', scalar.  Line length as a fraction of maximum length.
                Default is 0.5.
          - 'bean_median_color', MPL color.  Color of median marker.  Default
                is 'r'.
          - 'bean_median_marker', MPL marker.  Marker type, default is '+'.
          - 'jitter_marker', MPL marker.  Marker type for ``jitter=True``.
                Default is 'o'.
          - 'jitter_marker_size', int.  Marker size.  Default is 4.
          - 'jitter_fc', MPL color.  Jitter marker face color.  Default is None.
          - 'bean_legend_text', str.  If given, add a legend with given text.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    violinplot : Violin plot, also used internally in `beanplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    References
    ----------
    P. Kampstra, "Beanplot: A Boxplot Alternative for Visual Comparison of
    Distributions", J. Stat. Soft., Vol. 28, pp. 1-9, 2008.

    Examples
    --------
    We use the American National Election Survey 1996 dataset, which has Party
    Identification of respondents as independent variable and (among other
    data) age as dependent variable.

    >>> data = sm.datasets.anes96.load_pandas()
    >>> party_ID = np.arange(7)
    >>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
    ...           "Independent-Indpendent", "Independent-Republican",
    ...           "Weak Republican", "Strong Republican"]

    Group age by party ID, and create a violin plot with it:

    >>> plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible
    >>> age = [data.exog['age'][data.endog == id] for id in party_ID]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> sm.graphics.beanplot(age, ax=ax, labels=labels,
    ...                      plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
    ...                                 'label_fontsize':'small',
    ...                                 'label_rotation':30})
    >>> ax.set_xlabel("Party identification of respondent.")
    >>> ax.set_ylabel("Age")
    >>> plt.show()

    .. plot:: plots/graphics_boxplot_beanplot.py
    """
    pass

def _jitter_envelope(pos_data, xvals, violin, side):
    """Determine envelope for jitter markers."""
    pass

def _show_legend(ax):
    """Utility function to show legend."""
    pass