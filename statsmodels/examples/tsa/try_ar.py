"""
Created on Thu Oct 21 21:45:24 2010

Author: josef-pktd
"""
import numpy as np
from scipy import signal

def armaloop(arcoefs, macoefs, x):
    """get arma recursion in simple loop

    for simplicity assumes that ma polynomial is not longer than the ar-polynomial

    Parameters
    ----------
    arcoefs : array_like
        autoregressive coefficients in right hand side parameterization
    macoefs : array_like
        moving average coefficients, without leading 1

    Returns
    -------
    y : ndarray
        predicted values, initial values are the same as the observed values
    e : ndarray
        predicted residuals, zero for initial observations

    Notes
    -----
    Except for the treatment of initial observations this is the same as using
    scipy.signal.lfilter, which is much faster. Written for testing only
    """
    pass
arcoefs, macoefs = (-np.array([1, -0.8, 0.2])[1:], np.array([1.0, 0.5, 0.1])[1:])
print(armaloop(arcoefs, macoefs, np.ones(10)))
print(armaloop([0.8], [], np.ones(10)))
print(armaloop([0.8], [], np.arange(2, 10)))
y, e = armaloop([0.1], [0.8], np.arange(2, 10))
print(e)
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.arange(2, 10)))
y, e = armaloop([], [0.8], np.ones(10))
print(e)
print(signal.lfilter(np.array([1, -0.0]), np.array([1.0, 0.8]), np.ones(10)))
ic = signal.lfiltic(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones([0]), np.array([1]))
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones(10), zi=ic))
zi = signal.lfilter_zi(np.array([1, -0.8, 0.2]), np.array([1.0, 0, 0]))
print(signal.lfilter(np.array([1, -0.1]), np.array([1.0, 0.8]), np.ones(10), zi=zi))
print(signal.filtfilt(np.array([1, -0.8]), np.array([1.0]), np.ones(10)))