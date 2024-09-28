"""
Univariate Kernel Density Estimators

References
----------
Racine, Jeff. (2008) "Nonparametric Econometrics: A Primer," Foundation and
    Trends in Econometrics: Vol 3: No 1, pp1-88.
    http://dx.doi.org/10.1561/0800000009

https://en.wikipedia.org/wiki/Kernel_%28statistics%29

Silverman, B.W.  Density Estimation for Statistics and Data Analysis.
"""
import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin
kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov, uni=kernels.Uniform, tri=kernels.Triangular, biw=kernels.Biweight, triw=kernels.Triweight, cos=kernels.Cosine, cos2=kernels.Cosine2, tric=kernels.Tricube)

class KDEUnivariate:
    """
    Univariate Kernel Density Estimator.

    Parameters
    ----------
    endog : array_like
        The variable for which the density estimate is desired.

    Notes
    -----
    If cdf, sf, cumhazard, or entropy are computed, they are computed based on
    the definition of the kernel rather than the FFT approximation, even if
    the density is fit with FFT = True.

    `KDEUnivariate` is much faster than `KDEMultivariate`, due to its FFT-based
    implementation.  It should be preferred for univariate, continuous data.
    `KDEMultivariate` also supports mixed data.

    See Also
    --------
    KDEMultivariate
    kdensity, kdensityfft

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt

    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> dens = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
    >>> dens.fit()
    >>> plt.plot(dens.cdf)
    >>> plt.show()
    """

    def __init__(self, endog):
        self.endog = array_like(endog, 'endog', ndim=1, contiguous=True)

    def fit(self, kernel='gau', bw='normal_reference', fft=True, weights=None, gridsize=None, adjust=1, cut=3, clip=(-np.inf, np.inf)):
        """
        Attach the density estimate to the KDEUnivariate class.

        Parameters
        ----------
        kernel : str
            The Kernel to be used. Choices are:

            - "biw" for biweight
            - "cos" for cosine
            - "epa" for Epanechnikov
            - "gau" for Gaussian.
            - "tri" for triangular
            - "triw" for triweight
            - "uni" for uniform

        bw : str, float, callable
            The bandwidth to use. Choices are:

            - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
              `min(std(x),IQR/1.34)`
            - "silverman" - .9 * A * nobs ** (-1/5.), where A is
              `min(std(x),IQR/1.34)`
            - "normal_reference" - C * A * nobs ** (-1/5.), where C is
              calculated from the kernel. Equivalent (up to 2 dp) to the
              "scott" bandwidth for gaussian kernels. See bandwidths.py
            - If a float is given, its value is used as the bandwidth.
            - If a callable is given, it's return value is used.
              The callable should take exactly two parameters, i.e.,
              fn(x, kern), and return a float, where:

              * x - the clipped input data
              * kern - the kernel instance used

        fft : bool
            Whether or not to use FFT. FFT implementation is more
            computationally efficient. However, only the Gaussian kernel
            is implemented. If FFT is False, then a 'nobs' x 'gridsize'
            intermediate array is created.
        gridsize : int
            If gridsize is None, max(len(x), 50) is used.
        cut : float
            Defines the length of the grid past the lowest and highest values
            of x so that the kernel goes to zero. The end points are
            ``min(x) - cut * adjust * bw`` and ``max(x) + cut * adjust * bw``.
        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.

        Returns
        -------
        KDEUnivariate
            The instance fit,
        """
        pass

    @cache_readonly
    def cdf(self):
        """
        Returns the cumulative distribution function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
        pass

    @cache_readonly
    def cumhazard(self):
        """
        Returns the hazard function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
        pass

    @cache_readonly
    def sf(self):
        """
        Returns the survival function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
        pass

    @cache_readonly
    def entropy(self):
        """
        Returns the differential entropy evaluated at the support

        Notes
        -----
        Will not work if fit has not been called. 1e-12 is added to each
        probability to ensure that log(0) is not called.
        """
        pass

    @cache_readonly
    def icdf(self):
        """
        Inverse Cumulative Distribution (Quantile) Function

        Notes
        -----
        Will not work if fit has not been called. Uses
        `scipy.stats.mstats.mquantiles`.
        """
        pass

    def evaluate(self, point):
        """
        Evaluate density at a point or points.

        Parameters
        ----------
        point : {float, ndarray}
            Point(s) at which to evaluate the density.
        """
        pass

def kdensity(x, kernel='gau', bw='normal_reference', weights=None, gridsize=None, adjust=1, clip=(-np.inf, np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator.

    Parameters
    ----------
    x : array_like
        The variable for which the density estimate is desired.
    kernel : str
        The Kernel to be used. Choices are
        - "biw" for biweight
        - "cos" for cosine
        - "epa" for Epanechnikov
        - "gau" for Gaussian.
        - "tri" for triangular
        - "triw" for triweight
        - "uni" for uniform
    bw : str, float, callable
        The bandwidth to use. Choices are:

        - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "silverman" - .9 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "normal_reference" - C * A * nobs ** (-1/5.), where C is
          calculated from the kernel. Equivalent (up to 2 dp) to the
          "scott" bandwidth for gaussian kernels. See bandwidths.py
        - If a float is given, its value is used as the bandwidth.
        - If a callable is given, it's return value is used.
          The callable should take exactly two parameters, i.e.,
          fn(x, kern), and return a float, where:

          * x - the clipped input data
          * kern - the kernel instance used

    weights : array or None
        Optional  weights. If the x value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, max(len(x), 50) is used.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
    clip : tuple
        Observations in x that are outside of the range given by clip are
        dropped. The number of observations in x is then shortened.
    cut : float
        Defines the length of the grid past the lowest and highest values of x
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{min(x) or max(x)}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : ndarray
        The densities estimated at the grid points.
    grid : ndarray, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Creates an intermediate (`gridsize` x `nobs`) array. Use FFT for a more
    computationally efficient version.
    """
    pass

def kdensityfft(x, kernel='gau', bw='normal_reference', weights=None, gridsize=None, adjust=1, clip=(-np.inf, np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator

    Parameters
    ----------
    x : array_like
        The variable for which the density estimate is desired.
    kernel : str
        ONLY GAUSSIAN IS CURRENTLY IMPLEMENTED.
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gau" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
    bw : str, float, callable
        The bandwidth to use. Choices are:

        - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "silverman" - .9 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "normal_reference" - C * A * nobs ** (-1/5.), where C is
          calculated from the kernel. Equivalent (up to 2 dp) to the
          "scott" bandwidth for gaussian kernels. See bandwidths.py
        - If a float is given, its value is used as the bandwidth.
        - If a callable is given, it's return value is used.
          The callable should take exactly two parameters, i.e.,
          fn(x, kern), and return a float, where:

          * x - the clipped input data
          * kern - the kernel instance used

    weights : array or None
        WEIGHTS ARE NOT CURRENTLY IMPLEMENTED.
        Optional  weights. If the x value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, min(len(x), 512) is used. Note that the provided
        number is rounded up to the next highest power of 2.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
        clip : tuple
        Observations in x that are outside of the range given by clip are
        dropped. The number of observations in x is then shortened.
    cut : float
        Defines the length of the grid past the lowest and highest values of x
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{x.min() or x.max()}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : ndarray
        The densities estimated at the grid points.
    grid : ndarray, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Only the default kernel is implemented. Weights are not implemented yet.
    This follows Silverman (1982) with changes suggested by Jones and Lotwick
    (1984). However, the discretization step is replaced by linear binning
    of Fan and Marron (1994). This should be extended to accept the parts
    that are dependent only on the data to speed things up for
    cross-validation.

    References
    ----------
    Fan, J. and J.S. Marron. (1994) `Fast implementations of nonparametric
        curve estimators`. Journal of Computational and Graphical Statistics.
        3.1, 35-56.
    Jones, M.C. and H.W. Lotwick. (1984) `Remark AS R50: A Remark on Algorithm
        AS 176. Kernal Density Estimation Using the Fast Fourier Transform`.
        Journal of the Royal Statistical Society. Series C. 33.1, 120-2.
    Silverman, B.W. (1982) `Algorithm AS 176. Kernel density estimation using
        the Fast Fourier Transform. Journal of the Royal Statistical Society.
        Series C. 31.2, 93-9.
    """
    pass