from packaging.version import Version, parse
import numpy as np
import scipy
SP_VERSION = parse(scipy.__version__)
SP_LT_15 = SP_VERSION < Version('1.4.99')
SCIPY_GT_14 = not SP_LT_15
SP_LT_16 = SP_VERSION < Version('1.5.99')
SP_LT_17 = SP_VERSION < Version('1.6.99')
SP_LT_19 = SP_VERSION < Version('1.8.99')

def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    pass

def _valarray(shape, value=np.nan, typecode=None):
    """Return an array of all value."""
    pass
if SP_LT_16:
    from ._scipy_multivariate_t import multivariate_t
else:
    from scipy.stats import multivariate_t