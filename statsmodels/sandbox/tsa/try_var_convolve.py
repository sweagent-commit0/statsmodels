"""trying out VAR filtering and multidimensional fft

Note: second half is copy and paste and does not run as script
incomplete definitions of variables, some I created in shell

Created on Thu Jan 07 12:23:40 2010

Author: josef-pktd

update 2010-10-22
2 arrays were not defined, copied from fft_filter.log.py but I did not check
what the results are.
Runs now without raising exception
"""
import numpy as np
from numpy.testing import assert_equal
from scipy import signal, stats
try:
    from scipy.signal._signaltools import _centered as trim_centered
except ImportError:
    from scipy.signal.signaltools import _centered as trim_centered
from statsmodels.tsa.filters.filtertools import fftconvolveinv as fftconvolve
x = np.arange(40).reshape((2, 20)).T
x = np.arange(60).reshape((3, 20)).T
a3f = np.array([[[0.5, 1.0], [1.0, 0.5]], [[0.5, 1.0], [1.0, 0.5]]])
a3f = np.ones((2, 3, 3))
nlags = a3f.shape[0]
ntrim = nlags // 2
y0 = signal.convolve(x, a3f[:, :, 0], mode='valid')
y1 = signal.convolve(x, a3f[:, :, 1], mode='valid')
yf = signal.convolve(x[:, :, None], a3f)
y = yf[:, 1, :]
yvalid = yf[ntrim:-ntrim, yf.shape[1] // 2, :]
print(trim_centered(y, x.shape))
assert_equal(yvalid[:, 0], y0.ravel())
assert_equal(yvalid[:, 1], y1.ravel())

def arfilter(x, a):
    """apply an autoregressive filter to a series x

    x can be 2d, a can be 1d, 2d, or 3d

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    a : array_like
        autoregressive filter coefficients, ar lag polynomial
        see Notes

    Returns
    -------
    y : ndarray, 2d
        filtered array, number of columns determined by x and a

    Notes
    -----

    In general form this uses the linear filter ::

        y = a(L)x

    where
    x : nobs, nvars
    a : nlags, nvars, npoly

    Depending on the shape and dimension of a this uses different
    Lag polynomial arrays

    case 1 : a is 1d or (nlags,1)
        one lag polynomial is applied to all variables (columns of x)
    case 2 : a is 2d, (nlags, nvars)
        each series is independently filtered with its own
        lag polynomial, uses loop over nvar
    case 3 : a is 3d, (nlags, nvars, npoly)
        the ith column of the output array is given by the linear filter
        defined by the 2d array a[:,:,i], i.e. ::

            y[:,i] = a(.,.,i)(L) * x
            y[t,i] = sum_p sum_j a(p,j,i)*x(t-p,j)
                     for p = 0,...nlags-1, j = 0,...nvars-1,
                     for all t >= nlags


    Note: maybe convert to axis=1, Not

    TODO: initial conditions

    """
    pass
a3f = np.ones((2, 3, 3))
y0ar = arfilter(x, a3f[:, :, 0])
print(y0ar, x[1:] + x[:-1])
yres = arfilter(x, a3f[:, :, :2])
print(np.all(yres == (x[1:, :].sum(1) + x[:-1].sum(1))[:, None]))
yff = fftconvolve(x.astype(float)[:, :, None], a3f)
rvs = np.random.randn(500)
ar1fft = fftconvolve(rvs, np.array([1, -0.8]))
ar1fftp = fftconvolve(np.r_[np.zeros(100), rvs], np.array([1, -0.8]))
ar1lf = signal.lfilter([1], [1, -0.8], rvs)
ar1 = np.zeros(501)
for i in range(1, 501):
    ar1[i] = 0.8 * ar1[i - 1] + rvs[i - 1]
errar1 = np.zeros(501)
for i in range(1, 500):
    errar1[i] = rvs[i] - 0.8 * rvs[i - 1]
print('\n compare: \nerrloop - arloop - fft - lfilter - fftp (padded)')
print(np.column_stack((errar1[1:31], ar1[1:31], ar1fft[:30], ar1lf[:30], ar1fftp[100:130])))
print(maxabs(ar1[1:], ar1lf))
print(maxabs(ar1[1:], ar1fftp[100:-1]))
rvs3 = np.random.randn(500, 3)
a3n = np.array([[1, 1, 1], [-0.8, 0.5, 0.1]])
a3n = np.array([[1, 1, 1], [-0.8, 0.0, 0.0]])
a3n = np.array([[1, -1, -1], [-0.8, 0.0, 0.0]])
a3n = np.array([[1, 0, 0], [-0.8, 0.0, 0.0]])
a3ne = np.r_[np.ones((1, 3)), -0.8 * np.eye(3)]
a3ne = np.r_[np.ones((1, 3)), -0.8 * np.eye(3)]
ar13fft = fftconvolve(rvs3, a3n)
ar13 = np.zeros((501, 3))
for i in range(1, 501):
    ar13[i] = np.sum(a3n[1, :] * ar13[i - 1]) + rvs[i - 1]
imp = np.zeros((10, 3))
imp[0] = 1
a3n = np.array([[1, 0, 0], [-0.8, 0.0, 0.0]])
fftconvolve(np.r_[np.zeros((100, 3)), imp], a3n)[100:]
a3n = np.array([[1, 0, 0], [-0.8, -0.5, 0.0]])
fftconvolve(np.r_[np.zeros((100, 3)), imp], a3n)[100:]
a3n3 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[-0.8, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, -0.8]]])
a3n3 = np.array([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[-0.8, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, -0.8]]])
ttt = fftconvolve(np.r_[np.zeros((100, 3)), imp][:, :, None], a3n3.T)[100:]
gftt = ttt / ttt[0, :, :]
a3n3 = np.array([[[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[-0.8, 0.2, 0.0], [0, 0.0, 0.0], [0.0, 0.0, 0.8]]])
ttt = fftconvolve(np.r_[np.zeros((100, 3)), imp][:, :, None], a3n3)[100:]
gftt = ttt / ttt[0, :, :]
signal.fftconvolve(np.dstack((imp, imp, imp)), a3n3)[1, :, :]
nobs = 10
imp = np.zeros((nobs, 3))
imp[1] = 1.0
ar13 = np.zeros((nobs + 1, 3))
for i in range(1, nobs + 1):
    ar13[i] = np.dot(a3n3[1, :, :], ar13[i - 1]) + imp[i - 1]
a3n3inv = np.zeros((nobs + 1, 3, 3))
a3n3inv[0, :, :] = a3n3[0]
a3n3inv[1, :, :] = -a3n3[1]
for i in range(2, nobs + 1):
    a3n3inv[i, :, :] = np.dot(-a3n3[1], a3n3inv[i - 1, :, :])
a3n3sy = np.array([[[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[-0.8, 0.2, 0.0], [0, 0.0, 0.0], [0.0, 0.0, 0.8]]])
nobs = 10
a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.8, 0.0], [-0.1, -0.8]]])
a2n3inv = np.zeros((nobs + 1, 2, 2))
a2n3inv[0, :, :] = a[0]
a2n3inv[1, :, :] = -a[1]
for i in range(2, nobs + 1):
    a2n3inv[i, :, :] = np.dot(-a[1], a2n3inv[i - 1, :, :])
nobs = 10
imp = np.zeros((nobs, 2))
imp[0, 0] = 1.0
a2 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.8, 0.0], [0.1, -0.8]]])
ar12 = np.zeros((nobs + 1, 2))
for i in range(1, nobs + 1):
    ar12[i] = np.dot(-a2[1, :, :], ar12[i - 1]) + imp[i - 1]
u = np.random.randn(10, 2)
ar12r = np.zeros((nobs + 1, 2))
for i in range(1, nobs + 1):
    ar12r[i] = np.dot(-a2[1, :, :], ar12r[i - 1]) + u[i - 1]
a2inv = np.zeros((nobs + 1, 2, 2))
a2inv[0, :, :] = a2[0]
a2inv[1, :, :] = -a2[1]
for i in range(2, nobs + 1):
    a2inv[i, :, :] = np.dot(-a2[1], a2inv[i - 1, :, :])
nbins = 12
binProb = np.zeros(nbins) + 1.0 / nbins
binSumProb = np.add.accumulate(binProb)
print(binSumProb)
print(stats.gamma.ppf(binSumProb, 0.6379, loc=1.6, scale=39.555))