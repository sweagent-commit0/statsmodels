"""trying to verify theoretical acf of arma

explicit functions for autocovariance functions of ARIMA(1,1), MA(1), MA(2)
plus 3 functions from nitime.utils

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
ar = [1.0, -0.6]
ma = [1.0, 0.4]
mod = ''
x = arma_generate_sample(ar, ma, 5000)
x_acf = acf(x)[:10]
x_ir = arma_impulse_response(ar, ma)

def demean(x, axis=0):
    """Return x minus its mean along the specified axis"""
    pass

def detrend_mean(x):
    """Return x minus the mean(x)"""
    pass

def detrend_none(x):
    """Return x: no detrending"""
    pass

def detrend_linear(y):
    """Return y minus best fit line; 'linear' detrending """
    pass

def acovf_explicit(ar, ma, nobs):
    """add correlation of MA representation explicitely

    """
    pass
ar1 = [1.0, -0.8]
ar0 = [1.0, 0.0]
ma1 = [1.0, 0.4]
ma2 = [1.0, 0.4, 0.6]
ma0 = [1.0, 0.0]
comparefn = dict([('ma1', acovf_ma1), ('ma2', acovf_ma2), ('arma11', acovf_arma11), ('ar1', acovf_arma11)])
cases = [('ma1', (ar0, ma1)), ('ma2', (ar0, ma2)), ('arma11', (ar1, ma1)), ('ar1', (ar1, ma0))]
for c, args in cases:
    ar, ma = args
    print('')
    print(c, ar, ma)
    myacovf = arma_acovf(ar, ma, nobs=10)
    myacf = arma_acf(ar, ma, lags=10)
    if c[:2] == 'ma':
        othacovf = comparefn[c](ma)
    else:
        othacovf = comparefn[c](ar, ma)
    print(myacovf[:5])
    print(othacovf[:5])
    assert_array_almost_equal(myacovf, othacovf, 10)
    assert_array_almost_equal(myacf, othacovf / othacovf[0], 10)

def autocorr(s, axis=-1):
    """Returns the autocorrelation of signal s at all lags. Adheres to the
definition r(k) = E{s(n)s*(n-k)} where E{} is the expectation operator.
"""
    pass

def norm_corr(x, y, mode='valid'):
    """Returns the correlation between two ndarrays, by calling np.correlate in
'same' mode and normalizing the result by the std of the arrays and by
their lengths. This results in a correlation = 1 for an auto-correlation"""
    pass

def pltacorr(self, x, **kwargs):
    """
    call signature::

        acorr(x, normed=True, detrend=detrend_none, usevlines=True,
              maxlags=10, **kwargs)

    Plot the autocorrelation of *x*.  If *normed* = *True*,
    normalize the data by the autocorrelation at 0-th lag.  *x* is
    detrended by the *detrend* callable (default no normalization).

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length 2*maxlags+1 lag vector

      - *c* is the 2*maxlags+1 auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
        returned by :meth:`plot`

    The default *linestyle* is None and the default *marker* is
    ``'o'``, though these can be overridden with keyword args.
    The cross correlation is performed with
    :func:`numpy.correlate` with *mode* = 2.

    If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`
    rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw
    vertical lines from the origin to the acorr.  Otherwise, the
    plot style is determined by the kwargs, which are
    :class:`~matplotlib.lines.Line2D` properties.

    *maxlags* is a positive integer detailing the number of lags
    to show.  The default value of *None* will return all
    :math:`2 \\mathrm{len}(x) - 1` lags.

    The return value is a tuple (*lags*, *c*, *linecol*, *b*)
    where

    - *linecol* is the
      :class:`~matplotlib.collections.LineCollection`

    - *b* is the *x*-axis.

    .. seealso::

        :meth:`~matplotlib.axes.Axes.plot` or
        :meth:`~matplotlib.axes.Axes.vlines`
           For documentation on valid kwargs.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """
    pass

def pltxcorr(self, x, y, normed=True, detrend=detrend_none, usevlines=True, maxlags=10, **kwargs):
    """
    call signature::

        def xcorr(self, x, y, normed=True, detrend=detrend_none,
          usevlines=True, maxlags=10, **kwargs):

    Plot the cross correlation between *x* and *y*.  If *normed* =
    *True*, normalize the data by the cross correlation at 0-th
    lag.  *x* and y are detrended by the *detrend* callable
    (default no normalization).  *x* and *y* must be equal length.

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length ``2*maxlags+1`` lag vector

      - *c* is the ``2*maxlags+1`` auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
         returned by :func:`~matplotlib.pyplot.plot`.

    The default *linestyle* is *None* and the default *marker* is
    'o', though these can be overridden with keyword args.  The
    cross correlation is performed with :func:`numpy.correlate`
    with *mode* = 2.

    If *usevlines* is *True*:

       :func:`~matplotlib.pyplot.vlines`
       rather than :func:`~matplotlib.pyplot.plot` is used to draw
       vertical lines from the origin to the xcorr.  Otherwise the
       plotstyle is determined by the kwargs, which are
       :class:`~matplotlib.lines.Line2D` properties.

       The return value is a tuple (*lags*, *c*, *linecol*, *b*)
       where *linecol* is the
       :class:`matplotlib.collections.LineCollection` instance and
       *b* is the *x*-axis.

    *maxlags* is a positive integer detailing the number of lags to show.
    The default value of *None* will return all ``(2*len(x)-1)`` lags.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """
    pass
arrvs = ar_generator()
arma = ARIMA(arrvs[0])
res = arma.fit((4, 0, 0))
print(res[0])
acf1 = acf(arrvs[0])
acovf1b = acovf(arrvs[0], unbiased=False)
acf2 = autocorr(arrvs[0])
acf2m = autocorr(arrvs[0] - arrvs[0].mean())
print(acf1[:10])
print(acovf1b[:10])
print(acf2[:10])
print(acf2m[:10])
x = arma_generate_sample([1.0, -0.8], [1.0], 500)
print(acf(x)[:20])
print(regression.yule_walker(x, 10))
plt.plot(x)
plt.figure()
pltxcorr(plt, x, x)
plt.figure()
pltxcorr(plt, x, x, usevlines=False)
plt.figure()
plot_acf(plt, acf1[:20], np.arange(len(acf1[:20])), usevlines=True)
plt.figure()
ax = plt.subplot(211)
plot_acf(ax, acf1[:20], usevlines=True)
ax = plt.subplot(212)
plot_acf(ax, acf1[:20], np.arange(len(acf1[:20])), usevlines=False)