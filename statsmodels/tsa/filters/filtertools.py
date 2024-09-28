"""Linear Filters for time series analysis and testing


TODO:
* check common sequence in signature of filter functions (ar,ma,x) or (x,ar,ma)

Created on Sat Oct 23 17:18:03 2010

Author: Josef-pktd
"""
import numpy as np
import scipy.fftpack as fft
from scipy import signal
try:
    from scipy.signal._signaltools import _centered as trim_centered
except ImportError:
    from scipy.signal.signaltools import _centered as trim_centered
from statsmodels.tools.validation import array_like, PandasWrapper

def fftconvolveinv(in1, in2, mode='full'):
    """
    Convolve two N-dimensional arrays using FFT. See convolve.

    copied from scipy.signal.signaltools, but here used to try out inverse
    filter. does not work or I cannot get it to work

    2010-10-23:
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it does not work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    pass

def fftconvolve3(in1, in2=None, in3=None, mode='full'):
    """
    Convolve two N-dimensional arrays using FFT. See convolve.

    For use with arma  (old version: in1=num in2=den in3=data

    * better for consistency with other functions in1=data in2=num in3=den
    * note in2 and in3 need to have consistent dimension/shape
      since I'm using max of in2, in3 shapes and not the sum

    copied from scipy.signal.signaltools, but here used to try out inverse
    filter does not work or I cannot get it to work

    2010-10-23
    looks ok to me for 1d,
    from results below with padded data array (fftp)
    but it does not work for multidimensional inverse filter (fftn)
    original signal.fftconvolve also uses fftn
    """
    pass

def recursive_filter(x, ar_coeff, init=None):
    """
    Autoregressive, or recursive, filtering.

    Parameters
    ----------
    x : array_like
        Time-series data. Should be 1d or n x 1.
    ar_coeff : array_like
        AR coefficients in reverse time order. See Notes for details.
    init : array_like
        Initial values of the time-series prior to the first value of y.
        The default is zero.

    Returns
    -------
    array_like
        Filtered array, number of columns determined by x and ar_coeff. If x
        is a pandas object than a Series is returned.

    Notes
    -----
    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(n_coeff).
    """
    pass

def convolution_filter(x, filt, nsides=2):
    """
    Linear filtering via convolution. Centered and backward displaced moving
    weighted average.

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    filt : array_like
        Linear filter coefficients in reverse time-order. Should have the
        same number of dimensions as x though if 1d and ``x`` is 2d will be
        coerced to 2d.
    nsides : int, optional
        If 2, a centered moving average is computed using the filter
        coefficients. If 1, the filter coefficients are for past values only.
        Both methods use scipy.signal.convolve.

    Returns
    -------
    y : ndarray, 2d
        Filtered array, number of columns determined by x and filt. If a
        pandas object is given, a pandas object is returned. The index of
        the return is the exact same as the time period in ``x``

    Notes
    -----
    In nsides == 1, x is filtered ::

        y[n] = filt[0]*x[n-1] + ... + filt[n_filt-1]*x[n-n_filt]

    where n_filt is len(filt).

    If nsides == 2, x is filtered around lag 0 ::

        y[n] = filt[0]*x[n - n_filt/2] + ... + filt[n_filt / 2] * x[n]
               + ... + x[n + n_filt/2]

    where n_filt is len(filt). If n_filt is even, then more of the filter
    is forward in time than backward.

    If filt is 1d or (nlags,1) one lag polynomial is applied to all
    variables (columns of x). If filt is 2d, (nlags, nvars) each series is
    independently filtered with its own lag polynomial, uses loop over nvar.
    This is different than the usual 2d vs 2d convolution.

    Filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized data. For large data fft convolution would be
    faster.
    """
    pass

def miso_lfilter(ar, ma, x, useic=False):
    """
    Filter multiple time series into a single time series.

    Uses a convolution to merge inputs, and then lfilter to produce output.

    Parameters
    ----------
    ar : array_like
        The coefficients of autoregressive lag polynomial including lag zero,
        ar(L) in the expression ar(L)y_t.
    ma : array_like, same ndim as x, currently 2d
        The coefficient of the moving average lag polynomial, ma(L) in
        ma(L)x_t.
    x : array_like
        The 2-d input data series, time in rows, variables in columns.
    useic : bool
        Flag indicating whether to use initial conditions.

    Returns
    -------
    y : ndarray
        The filtered output series.
    inp : ndarray, 1d
        The combined input series.

    Notes
    -----
    currently for 2d inputs only, no choice of axis
    Use of signal.lfilter requires that ar lag polynomial contains
    floating point numbers
    does not cut off invalid starting and final values

    miso_lfilter find array y such that:

            ar(L)y_t = ma(L)x_t

    with shapes y (nobs,), x (nobs, nvars), ar (narlags,), and
    ma (narlags, nvars).
    """
    pass