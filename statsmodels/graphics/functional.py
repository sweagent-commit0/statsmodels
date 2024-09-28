"""Module for functional boxplots."""
from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
try:
    from scipy.optimize import brute, differential_evolution, fmin
    have_de_optim = True
except ImportError:
    from scipy.optimize import brute, fmin
    have_de_optim = False
import itertools
from multiprocessing import Pool
from . import utils
__all__ = ['hdrboxplot', 'fboxplot', 'rainbowplot', 'banddepth']

class HdrResults:
    """Wrap results and pretty print them."""

    def __init__(self, kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        msg = 'HDR boxplot summary:\n-> median:\n{}\n-> 50% HDR (max, min):\n{}\n-> 90% HDR (max, min):\n{}\n-> Extra quantiles (max, min):\n{}\n-> Outliers:\n{}\n-> Outliers indices:\n{}\n'.format(self.median, self.hdr_50, self.hdr_90, self.extra_quantiles, self.outliers, self.outliers_idx)
        return msg

def _inverse_transform(pca, data):
    """
    Inverse transform on PCA.

    Use PCA's `project` method by temporary replacing its factors with
    `data`.

    Parameters
    ----------
    pca : statsmodels Principal Component Analysis instance
        The PCA object to use.
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.

    Returns
    -------
    projection : ndarray
        nobs by nvar array of the projection onto ncomp factors
    """
    pass

def _curve_constrained(x, idx, sign, band, pca, ks_gaussian):
    """Find out if the curve is within the band.

    The curve value at :attr:`idx` for a given PDF is only returned if
    within bounds defined by the band. Otherwise, 1E6 is returned.

    Parameters
    ----------
    x : float
        Curve in reduced space.
    idx : int
        Index value of the components to compute.
    sign : int
        Return positive or negative value.
    band : list of float
        PDF values `[min_pdf, max_pdf]` to be within.
    pca : statsmodels Principal Component Analysis instance
        The PCA object to use.
    ks_gaussian : KDEMultivariate instance

    Returns
    -------
    value : float
        Curve value at `idx`.
    """
    pass

def _min_max_band(args):
    """
    Min and max values at `idx`.

    Global optimization to find the extrema per component.

    Parameters
    ----------
    args: list
        It is a list of an idx and other arguments as a tuple:
            idx : int
                Index value of the components to compute
        The tuple contains:
            band : list of float
                PDF values `[min_pdf, max_pdf]` to be within.
            pca : statsmodels Principal Component Analysis instance
                The PCA object to use.
            bounds : sequence
                ``(min, max)`` pair for each components
            ks_gaussian : KDEMultivariate instance

    Returns
    -------
    band : tuple of float
        ``(max, min)`` curve values at `idx`
    """
    pass

def hdrboxplot(data, ncomp=2, alpha=None, threshold=0.95, bw=None, xdata=None, labels=None, ax=None, use_brute=False, seed=None):
    """
    High Density Region boxplot

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    ncomp : int, optional
        Number of components to use.  If None, returns the as many as the
        smaller of the number of rows or columns in data.
    alpha : list of floats between 0 and 1, optional
        Extra quantile values to compute. Default is None
    threshold : float between 0 and 1, optional
        Percentile threshold value for outliers detection. High value means
        a lower sensitivity to outliers. Default is `0.95`.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth. If `None`, set to
        `normal_reference`. If a string, should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    xdata : ndarray, optional
        The independent variable for the data. If not given, it is assumed to
        be an array of integers 0..N-1 with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`. If not given,
        outliers are labeled in the plot with array indices.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    use_brute : bool
        Use the brute force optimizer instead of the default differential
        evolution to find the curves. Default is False.
    seed : {None, int, np.random.RandomState}
        Seed value to pass to scipy.optimize.differential_evolution. Can be an
        integer or RandomState instance. If None, then the default RandomState
        provided by np.random is used.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    hdr_res : HdrResults instance
        An `HdrResults` instance with the following attributes:

         - 'median', array. Median curve.
         - 'hdr_50', array. 50% quantile band. [sup, inf] curves
         - 'hdr_90', list of array. 90% quantile band. [sup, inf]
            curves.
         - 'extra_quantiles', list of array. Extra quantile band.
            [sup, inf] curves.
         - 'outliers', ndarray. Outlier curves.

    See Also
    --------
    banddepth, rainbowplot, fboxplot

    Notes
    -----
    The median curve is the curve with the highest probability on the reduced
    space of a Principal Component Analysis (PCA).

    Outliers are defined as curves that fall outside the band corresponding
    to the quantile given by `threshold`.

    The non-outlying region is defined as the band made up of all the
    non-outlying curves.

    Behind the scene, the dataset is represented as a matrix. Each line
    corresponding to a 1D curve. This matrix is then decomposed using Principal
    Components Analysis (PCA). This allows to represent the data using a finite
    number of modes, or components. This compression process allows to turn the
    functional representation into a scalar representation of the matrix. In
    other words, you can visualize each curve from its components. Each curve
    is thus a point in this reduced space. With 2 components, this is called a
    bivariate plot (2D plot).

    In this plot, if some points are adjacent (similar components), it means
    that back in the original space, the curves are similar. Then, finding the
    median curve means finding the higher density region (HDR) in the reduced
    space. Moreover, the more you get away from this HDR, the more the curve is
    unlikely to be similar to the other curves.

    Using a kernel smoothing technique, the probability density function (PDF)
    of the multivariate space can be recovered. From this PDF, it is possible
    to compute the density probability linked to the cluster of points and plot
    its contours.

    Finally, using these contours, the different quantiles can be extracted
    along with the median curve and the outliers.

    Steps to produce the HDR boxplot include:

    1. Compute a multivariate kernel density estimation
    2. Compute contour lines for quantiles 90%, 50% and `alpha` %
    3. Plot the bivariate plot
    4. Compute median curve along with quantiles and outliers curves.

    References
    ----------
    [1] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-45, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a functional boxplot.  We see that the years 1982-83 and 1997-98 are
    outliers; these are the years where El Nino (a climate pattern
    characterized by warming up of the sea surface and higher air pressures)
    occurred with unusual intensity.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.hdrboxplot(data.raw_data[:, 1:],
    ...                              labels=data.raw_data[:, 0].astype(int),
    ...                              ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    .. plot:: plots/graphics_functional_hdrboxplot.py
    """
    pass

def fboxplot(data, xdata=None, labels=None, depth=None, method='MBD', wfactor=1.5, ax=None, plot_opts=None):
    """
    Plot functional boxplot.

    A functional boxplot is the analog of a boxplot for functional data.
    Functional data is any type of data that varies over a continuum, i.e.
    curves, probability distributions, seasonal data, etc.

    The data is first ordered, the order statistic used here is `banddepth`.
    Plotted are then the median curve, the envelope of the 50% central region,
    the maximum non-outlying envelope and the outlier curves.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N-1 with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`.  If given, outliers
        are labeled in the plot.
    depth : ndarray, optional
        A 1-D array of band depths for `data`, or equivalent order statistic.
        If not given, it will be calculated through `banddepth`.
    method : {'MBD', 'BD2'}, optional
        The method to use to calculate the band depth.  Default is 'MBD'.
    wfactor : float, optional
        Factor by which the central 50% region is multiplied to find the outer
        region (analog of "whiskers" of a classical boxplot).
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'cmap_outliers', a Matplotlib LinearSegmentedColormap instance.
          - 'c_inner', valid MPL color. Color of the central 50% region
          - 'c_outer', valid MPL color. Color of the non-outlying region
          - 'c_median', valid MPL color. Color of the median.
          - 'lw_outliers', scalar.  Linewidth for drawing outlier curves.
          - 'lw_median', scalar.  Linewidth for drawing the median curve.
          - 'draw_nonout', bool.  If True, also draw non-outlying curves.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    depth : ndarray
        A 1-D array containing the calculated band depths of the curves.
    ix_depth : ndarray
        A 1-D array of indices needed to order curves (or `depth`) from most to
        least central curve.
    ix_outliers : ndarray
        A 1-D array of indices of outlying curves in `data`.

    See Also
    --------
    banddepth, rainbowplot

    Notes
    -----
    The median curve is the curve with the highest band depth.

    Outliers are defined as curves that fall outside the band created by
    multiplying the central region by `wfactor`.  Note that the range over
    which they fall outside this band does not matter, a single data point
    outside the band is enough.  If the data is noisy, smoothing may therefore
    be required.

    The non-outlying region is defined as the band made up of all the
    non-outlying curves.

    References
    ----------
    [1] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of Computational
        and Graphical Statistics, vol. 20, pp. 1-19, 2011.
    [2] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-45, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a functional boxplot.  We see that the years 1982-83 and 1997-98 are
    outliers; these are the years where El Nino (a climate pattern
    characterized by warming up of the sea surface and higher air pressures)
    occurred with unusual intensity.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.fboxplot(data.raw_data[:, 1:], wfactor=2.58,
    ...                            labels=data.raw_data[:, 0].astype(int),
    ...                            ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    .. plot:: plots/graphics_functional_fboxplot.py
    """
    pass

def rainbowplot(data, xdata=None, depth=None, method='MBD', ax=None, cmap=None):
    """
    Create a rainbow plot for a set of curves.

    A rainbow plot contains line plots of all curves in the dataset, colored in
    order of functional depth.  The median curve is shown in black.

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    xdata : ndarray, optional
        The independent variable for the data.  If not given, it is assumed to
        be an array of integers 0..N-1 with N the length of the vectors in
        `data`.
    depth : ndarray, optional
        A 1-D array of band depths for `data`, or equivalent order statistic.
        If not given, it will be calculated through `banddepth`.
    method : {'MBD', 'BD2'}, optional
        The method to use to calculate the band depth.  Default is 'MBD'.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    cmap : Matplotlib LinearSegmentedColormap instance, optional
        The colormap used to color curves with.  Default is a rainbow colormap,
        with red used for the most central and purple for the least central
        curves.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    banddepth, fboxplot

    References
    ----------
    [1] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-25, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a rainbow plot:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.rainbowplot(data.raw_data[:, 1:], ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])
    >>> plt.show()

    .. plot:: plots/graphics_functional_rainbowplot.py
    """
    pass

def banddepth(data, method='MBD'):
    """
    Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters
    ----------
    data : ndarray
        The vectors of functions to create a functional boxplot from.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    method : {'MBD', 'BD2'}, optional
        Whether to use the original band depth (with J=2) of [1]_ or the
        modified band depth.  See Notes for details.

    Returns
    -------
    ndarray
        Depth values for functional curves.

    Notes
    -----
    Functional band depth as an order statistic for functional data was
    proposed in [1]_ and applied to functional boxplots and bagplots in [2]_.

    The method 'BD2' checks for each curve whether it lies completely inside
    bands constructed from two curves.  All permutations of two curves in the
    set of curves are used, and the band depth is normalized to one.  Due to
    the complete curve having to fall within the band, this method yields a lot
    of ties.

    The method 'MBD' is similar to 'BD2', but checks the fraction of the curve
    falling within the bands.  It therefore generates very few ties.

    The algorithm uses the efficient implementation proposed in [3]_.

    References
    ----------
    .. [1] S. Lopez-Pintado and J. Romo, "On the Concept of Depth for
           Functional Data", Journal of the American Statistical Association,
           vol.  104, pp. 718-734, 2009.
    .. [2] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of
           Computational and Graphical Statistics, vol. 20, pp. 1-19, 2011.
    .. [3] Y. Sun, M. G. Gentonb and D. W. Nychkac, "Exact fast computation
           of band depth for large functional datasets: How quickly can one
           million curves be ranked?", Journal for the Rapid Dissemination
           of Statistics Research, vol. 1, pp. 68-74, 2012.
    """
    pass