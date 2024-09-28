"""
This models contains the Kernels for Kernel smoothing.

Hopefully in the future they may be reused/extended for other kernel based
method

References:
----------

Pointwise Kernel Confidence Bounds
(smoothconf)
http://fedc.wiwi.hu-berlin.de/xplore/ebooks/html/anr/anrhtmlframe62.html
"""
from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf

class NdKernel:
    """Generic N-dimensial kernel

    Parameters
    ----------
    n : int
        The number of series for kernel estimates
    kernels : list
        kernels

    Can be constructed from either
    a) a list of n kernels which will be treated as
    indepent marginals on a gaussian copula (specified by H)
    or b) a single univariate kernel which will be applied radially to the
    mahalanobis distance defined by H.

    In the case of the Gaussian these are both equivalent, and the second constructiong
    is prefered.
    """

    def __init__(self, n, kernels=None, H=None):
        if kernels is None:
            kernels = Gaussian()
        self._kernels = kernels
        self.weights = None
        if H is None:
            H = np.matrix(np.identity(n))
        self._H = H
        self._Hrootinv = np.linalg.cholesky(H.I)

    def getH(self):
        """Getter for kernel bandwidth, H"""
        pass

    def setH(self, value):
        """Setter for kernel bandwidth, H"""
        pass
    H = property(getH, setH, doc='Kernel bandwidth matrix')

    def _kernweight(self, x):
        """returns the kernel weight for the independent multivariate kernel"""
        pass

    def __call__(self, x):
        """
        This simply returns the value of the kernel function at x

        Does the same as weight if the function is normalised
        """
        return self._kernweight(x)

class CustomKernel:
    """
    Generic 1D Kernel object.
    Can be constructed by selecting a standard named Kernel,
    or providing a lambda expression and domain.
    The domain allows some algorithms to run faster for finite domain kernels.
    """

    def __init__(self, shape, h=1.0, domain=None, norm=None):
        """
        shape should be a function taking and returning numeric type.

        For sanity it should always return positive or zero but this is not
        enforced in case you want to do weird things. Bear in mind that the
        statistical tests etc. may not be valid for non-positive kernels.

        The bandwidth of the kernel is supplied as h.

        You may specify a domain as a list of 2 values [min, max], in which case
        kernel will be treated as zero outside these values. This will speed up
        calculation.

        You may also specify the normalisation constant for the supplied Kernel.
        If you do this number will be stored and used as the normalisation
        without calculation.  It is recommended you do this if you know the
        constant, to speed up calculation.  In particular if the shape function
        provided is already normalised you should provide norm = 1.0.

        Warning: I think several calculations assume that the kernel is
        normalized. No tests for non-normalized kernel.
        """
        self._normconst = norm
        self.domain = domain
        self.weights = None
        if callable(shape):
            self._shape = shape
        else:
            raise TypeError('shape must be a callable object/function')
        self._h = h
        self._L2Norm = None
        self._kernel_var = None
        self._normal_reference_constant = None
        self._order = None

    def geth(self):
        """Getter for kernel bandwidth, h"""
        pass

    def seth(self, value):
        """Setter for kernel bandwidth, h"""
        pass
    h = property(geth, seth, doc='Kernel Bandwidth')

    def in_domain(self, xs, ys, x):
        """
        Returns the filtered (xs, ys) based on the Kernel domain centred on x
        """
        pass

    def density(self, xs, x):
        """Returns the kernel density estimate for point x based on x-values
        xs
        """
        pass

    def density_var(self, density, nobs):
        """approximate pointwise variance for kernel density

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        kde_var : ndarray
            estimated variance of the density estimate

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate.
        """
        pass

    def density_confint(self, density, nobs, alpha=0.05):
        """approximate pointwise confidence interval for kernel density

        The confidence interval is centered at the estimated density and
        ignores the bias of the density estimate.

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        conf_int : ndarray
            estimated confidence interval of the density estimate, lower bound
            in first column and upper bound in second column

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate. The lower bound can be negative for density
        values close to zero.
        """
        pass

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.
        """
        pass

    def smoothvar(self, xs, ys, x):
        """Returns the kernel smoothing estimate of the variance at point x.
        """
        pass

    def smoothconf(self, xs, ys, x, alpha=0.05):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        pass

    @property
    def L2Norm(self):
        """Returns the integral of the square of the kernal from -inf to inf"""
        pass

    @property
    def norm_const(self):
        """
        Normalising constant for kernel (integral from -inf to inf)
        """
        pass

    @property
    def kernel_var(self):
        """Returns the second moment of the kernel"""
        pass

    @property
    def normal_reference_constant(self):
        """
        Constant used for silverman normal reference asymtotic bandwidth
        calculation.

        C  = 2((pi^(1/2)*(nu!)^3 R(k))/(2nu(2nu)!kap_nu(k)^2))^(1/(2nu+1))
        nu = kernel order
        kap_nu = nu'th moment of kernel
        R = kernel roughness (square of L^2 norm)

        Note: L2Norm property returns square of norm.
        """
        pass

    def weight(self, x):
        """This returns the normalised weight at distance x"""
        pass

    def __call__(self, x):
        """
        This simply returns the value of the kernel function at x

        Does the same as weight if the function is normalised
        """
        return self._shape(x)

class Uniform(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.5 * np.ones(x.shape), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 0.5
        self._kernel_var = 1.0 / 3
        self._order = 2

class Triangular(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 - abs(x), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 2.0 / 3.0
        self._kernel_var = 1.0 / 6
        self._order = 2

class Epanechnikov(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.75 * (1 - x * x), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 0.6
        self._kernel_var = 0.2
        self._order = 2

class Biweight(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.9375 * (1 - x * x) ** 2, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 5.0 / 7.0
        self._kernel_var = 1.0 / 7
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimized for Biweight.
        """
        pass

    def smoothvar(self, xs, ys, x):
        """
        Returns the kernel smoothing estimate of the variance at point x.
        """
        pass

    def smoothconf_(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        pass

class Triweight(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1.09375 * (1 - x * x) ** 3, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 350.0 / 429.0
        self._kernel_var = 1.0 / 9
        self._order = 2

class Gaussian(CustomKernel):
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.3989422804014327 * np.exp(-x ** 2 / 2.0), h=h, domain=None, norm=1.0)
        self._L2Norm = 1.0 / (2.0 * np.sqrt(np.pi))
        self._kernel_var = 1.0
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimized for Gaussian.
        """
        pass

class Cosine(CustomKernel):
    """
    Cosine Kernel

    K(u) = pi/4 cos(0.5 * pi * u) between -1.0 and 1.0
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.7853981633974483 * np.cos(np.pi / 2.0 * x), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = np.pi ** 2 / 16.0
        self._kernel_var = 0.1894305308612978
        self._order = 2

class Cosine2(CustomKernel):
    """
    Cosine2 Kernel

    K(u) = 1 + cos(2 * pi * u) between -0.5 and 0.5

    Note: this  is the same Cosine kernel that Stata uses
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 + np.cos(2.0 * np.pi * x), h=h, domain=[-0.5, 0.5], norm=1.0)
        self._L2Norm = 1.5
        self._kernel_var = 0.03267274151216444
        self._order = 2

class Tricube(CustomKernel):
    """
    Tricube Kernel

    K(u) = 0.864197530864 * (1 - abs(x)**3)**3 between -1.0 and 1.0
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.864197530864 * (1 - abs(x) ** 3) ** 3, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 175.0 / 247.0
        self._kernel_var = 35.0 / 243.0
        self._order = 2