"""
This module contains scatterplot smoothers, that is classes
who generate a smooth fit of a set of (x,y) pairs.
"""
import numpy as np
from . import kernels

class KernelSmoother:
    """
    1D Kernel Density Regression/Kernel Smoother

    Requires:
    x - array_like of x values
    y - array_like of y values
    Kernel - Kernel object, Default is Gaussian.
    """

    def __init__(self, x, y, Kernel=None):
        if Kernel is None:
            Kernel = kernels.Gaussian()
        self.Kernel = Kernel
        self.x = np.array(x)
        self.y = np.array(y)

    def __call__(self, x):
        return np.array([self.predict(xx) for xx in x])

    def predict(self, x):
        """
        Returns the kernel smoothed prediction at x

        If x is a real number then a single value is returned.

        Otherwise an attempt is made to cast x to numpy.ndarray and an array of
        corresponding y-points is returned.
        """
        pass

    def conf(self, x):
        """
        Returns the fitted curve and 1-sigma upper and lower point-wise
        confidence.
        These bounds are based on variance only, and do not include the bias.
        If the bandwidth is much larger than the curvature of the underlying
        function then the bias could be large.

        x is the points on which you want to evaluate the fit and the errors.

        Alternatively if x is specified as a positive integer, then the fit and
        confidence bands points will be returned after every
        xth sample point - so they are closer together where the data
        is denser.
        """
        pass

class PolySmoother:
    """
    Polynomial smoother up to a given order.
    Fit based on weighted least squares.

    The x values can be specified at instantiation or when called.

    This is a 3 liner with OLS or WLS, see test.
    It's here as a test smoother for GAM
    """

    def __init__(self, order, x=None):
        self.order = order
        self.coef = np.zeros((order + 1,), np.float64)
        if x is not None:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother init, shape:', x.shape)
                x = x[0, :]
            self.X = np.array([x ** i for i in range(order + 1)]).T

    def df_fit(self):
        """alias of df_model for backwards compatibility
        """
        pass

    def df_model(self):
        """
        Degrees of freedom used in the fit.
        """
        pass

    def smooth(self, *args, **kwds):
        """alias for fit,  for backwards compatibility,

        do we need it with different behavior than fit?

        """
        pass

    def df_resid(self):
        """
        Residual degrees of freedom from last fit.
        """
        pass

    def __call__(self, x=None):
        return self.predict(x=x)