"""get versions of mstats percentile functions that also work with non-masked arrays

uses dispatch to mstats version for difficult cases:
  - data is masked array
  - data requires nan handling (masknan=True)
  - data should be trimmed (limit is non-empty)
handle simple cases directly, which does not require apply_along_axis
changes compared to mstats: plotting_positions for n-dim with axis argument
addition: plotting_positions_w1d: with weights, 1d ndarray only

TODO:
consistency with scipy.stats versions not checked
docstrings from mstats not updated yet
code duplication, better solutions (?)
convert examples to tests
rename alphap, betap for consistency
timing question: one additional argsort versus apply_along_axis
weighted plotting_positions
- I have not figured out nd version of weighted plotting_positions
- add weighted quantiles


"""
import numpy as np
from numpy import ma
from scipy import stats

def quantiles(a, prob=list([0.25, 0.5, 0.75]), alphap=0.4, betap=0.4, axis=None, limit=(), masknan=False):
    """
    Computes empirical quantiles for a data array.

    Samples quantile are defined by :math:`Q(p) = (1-g).x[i] +g.x[i+1]`,
    where :math:`x[j]` is the *j*th order statistic, and
    `i = (floor(n*p+m))`, `m=alpha+p*(1-alpha-beta)` and `g = n*p + m - i`.

    Typical values of (alpha,beta) are:
        - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
        - (.5,.5)  : *p(k) = (k+1/2.)/n* : piecewise linear
          function (R, type 5)
        - (0,0)    : *p(k) = k/(n+1)* : (R type 6)
        - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
          That's R default (R type 7)
        - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM ?? JP
        - (0.35, 0.65): PWM   ?? JP  p(k) = (k-0.35)/n

    Parameters
    ----------
    a : array_like
        Input data, as a sequence or array of dimension at most 2.
    prob : array_like, optional
        List of quantiles to compute.
    alpha : float, optional
        Plotting positions parameter, default is 0.4.
    beta : float, optional
        Plotting positions parameter, default is 0.4.
    axis : int, optional
        Axis along which to perform the trimming.
        If None (default), the input array is first flattened.
    limit : tuple
        Tuple of (lower, upper) values.
        Values of `a` outside this closed interval are ignored.

    Returns
    -------
    quants : MaskedArray
        An array containing the calculated quantiles.

    Examples
    --------
    >>> from scipy.stats.mstats import mquantiles
    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    >>> mquantiles(a)
    array([ 19.2,  40. ,  42.8])

    Using a 2D array, specifying axis and limit.

    >>> data = np.array([[   6.,    7.,    1.],
                         [  47.,   15.,    2.],
                         [  49.,   36.,    3.],
                         [  15.,   39.,    4.],
                         [  42.,   40., -999.],
                         [  41.,   41., -999.],
                         [   7., -999., -999.],
                         [  39., -999., -999.],
                         [  43., -999., -999.],
                         [  40., -999., -999.],
                         [  36., -999., -999.]])
    >>> mquantiles(data, axis=0, limit=(0, 50))
    array([[ 19.2 ,  14.6 ,   1.45],
           [ 40.  ,  37.5 ,   2.5 ],
           [ 42.8 ,  40.05,   3.55]])

    >>> data[:, 2] = -999.
    >>> mquantiles(data, axis=0, limit=(0, 50))
    masked_array(data =
     [[19.2 14.6 --]
     [40.0 37.5 --]
     [42.8 40.05 --]],
                 mask =
     [[False False  True]
      [False False  True]
      [False False  True]],
           fill_value = 1e+20)
    """
    pass

def scoreatpercentile(data, per, limit=(), alphap=0.4, betap=0.4, axis=0, masknan=None):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantile
    """
    pass

def plotting_positions(data, alpha=0.4, beta=0.4, axis=0, masknan=False):
    """Returns the plotting positions (or empirical percentile points) for the
    data.
    Plotting positions are defined as (i-alpha)/(n+1-alpha-beta), where:
        - i is the rank order statistics (starting at 1)
        - n is the number of unmasked values along the given axis
        - alpha and beta are two parameters.

    Typical values for alpha and beta are:
        - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
        - (.5,.5)  : *p(k) = (k-1/2.)/n* : piecewise linear function (R, type 5)
          (Bliss 1967: "Rankit")
        - (0,0)    : *p(k) = k/(n+1)* : Weibull (R type 6), (Van der Waerden 1952)
        - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
          That's R default (R type 7)
        - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8), (Tukey 1962)
        - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9) (Blom 1958)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    Parameters
    ----------
    x : sequence
        Input data, as a sequence or array of dimension at most 2.
    prob : sequence
        List of quantiles to compute.
    alpha : {0.4, float} optional
        Plotting positions parameter.
    beta : {0.4, float} optional
        Plotting positions parameter.

    Notes
    -----
    I think the adjustments assume that there are no ties in order to be a reasonable
    approximation to a continuous density function. TODO: check this

    References
    ----------
    unknown,
    dates to original papers from Beasley, Erickson, Allison 2009 Behav Genet
    """
    pass
meppf = plotting_positions

def plotting_positions_w1d(data, weights=None, alpha=0.4, beta=0.4, method='notnormed'):
    """Weighted plotting positions (or empirical percentile points) for the data.

    observations are weighted and the plotting positions are defined as
    (ws-alpha)/(n-alpha-beta), where:
        - ws is the weighted rank order statistics or cumulative weighted sum,
          normalized to n if method is "normed"
        - n is the number of values along the given axis if method is "normed"
          and total weight otherwise
        - alpha and beta are two parameters.

    wtd.quantile in R package Hmisc seems to use the "notnormed" version.
    notnormed coincides with unweighted segment in example, drop "normed" version ?


    See Also
    --------
    plotting_positions : unweighted version that works also with more than one
        dimension and has other options
    """
    pass

def edf_normal_inverse_transformed(x, alpha=3.0 / 8, beta=3.0 / 8, axis=0):
    """rank based normal inverse transformed cdf
    """
    pass
if __name__ == '__main__':
    x = np.arange(5)
    print(plotting_positions(x))
    x = np.arange(10).reshape(-1, 2)
    print(plotting_positions(x))
    print(quantiles(x, axis=0))
    print(quantiles(x, axis=None))
    print(quantiles(x, axis=1))
    xm = ma.array(x)
    x2 = x.astype(float)
    x2[1, 0] = np.nan
    print(plotting_positions(xm, axis=0))
    for sl1 in [slice(None), 0]:
        print((plotting_positions(xm[sl1, 0]) == plotting_positions(x[sl1, 0])).all())
        print((quantiles(xm[sl1, 0]) == quantiles(x[sl1, 0])).all())
        print((stats.mstats.mquantiles(ma.fix_invalid(x2[sl1, 0])) == quantiles(x2[sl1, 0], masknan=1)).all())
    for ax in [0, 1, None, -1]:
        print((plotting_positions(xm, axis=ax) == plotting_positions(x, axis=ax)).all())
        print((quantiles(xm, axis=ax) == quantiles(x, axis=ax)).all())
        print((stats.mstats.mquantiles(ma.fix_invalid(x2), axis=ax) == quantiles(x2, axis=ax, masknan=1)).all())
    print((stats.mstats.plotting_positions(ma.fix_invalid(x2)) == plotting_positions(x2, axis=None, masknan=1)).all())
    x3 = np.dstack((x, x)).T
    for ax in [1, 2]:
        print((plotting_positions(x3, axis=ax)[0] == plotting_positions(x.T, axis=ax - 1)).all())
    np.testing.assert_equal(plotting_positions(np.arange(10), alpha=0.35, beta=1 - 0.35), (1 + np.arange(10) - 0.35) / 10)
    np.testing.assert_equal(plotting_positions(np.arange(10), alpha=0.4, beta=0.4), (1 + np.arange(10) - 0.4) / (10 + 0.2))
    np.testing.assert_equal(plotting_positions(np.arange(10)), (1 + np.arange(10) - 0.4) / (10 + 0.2))
    print('')
    print(scoreatpercentile(x, [10, 90]))
    print(plotting_positions_w1d(x[:, 0]))
    print((plotting_positions_w1d(x[:, 0]) == plotting_positions(x[:, 0])).all())
    w1 = [1, 1, 2, 1, 1]
    plotexample = 1
    if plotexample:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('ppf, cdf values on horizontal axis')
        plt.step(plotting_positions_w1d(x[:, 0], weights=w1, method='0'), x[:, 0], where='post')
        plt.step(stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), np.repeat(x[:, 0], w1, axis=0), where='post')
        plt.plot(plotting_positions_w1d(x[:, 0], weights=w1, method='0'), x[:, 0], '-o')
        plt.plot(stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), np.repeat(x[:, 0], w1, axis=0), '-o')
        plt.figure()
        plt.title('cdf, cdf values on vertical axis')
        plt.step(x[:, 0], plotting_positions_w1d(x[:, 0], weights=w1, method='0'), where='post')
        plt.step(np.repeat(x[:, 0], w1, axis=0), stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), where='post')
        plt.plot(x[:, 0], plotting_positions_w1d(x[:, 0], weights=w1, method='0'), '-o')
        plt.plot(np.repeat(x[:, 0], w1, axis=0), stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), '-o')
    plt.show()