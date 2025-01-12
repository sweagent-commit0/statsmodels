"""using scipy signal and numpy correlate to calculate some time series
statistics

original developer notes

see also scikits.timeseries  (movstat is partially inspired by it)
added 2009-08-29
timeseries moving stats are in c, autocorrelation similar to here
I thought I saw moving stats somewhere in python, maybe not)


TODO

moving statistics
- filters do not handle boundary conditions nicely (correctly ?)
e.g. minimum order filter uses 0 for out of bounds value
-> append and prepend with last resp. first value
- enhance for nd arrays, with axis = 0



Note: Equivalence for 1D signals
>>> np.all(signal.correlate(x,[1,1,1],'valid')==np.correlate(x,[1,1,1]))
True
>>> np.all(ndimage.filters.correlate(x,[1,1,1], origin = -1)[:-3+1]==np.correlate(x,[1,1,1]))
True

# multidimensional, but, it looks like it uses common filter across time series, no VAR
ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)
ndimage.filters.correlate(x,[1,1,1],origin = 1))
ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), origin = 1)

>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)[0]==ndimage.filters.correlate(x,[1,1,1],origin = 1))
True
>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), origin = 1)[0]==ndimage.filters.correlate(x,[1,1,1],origin = 1))


update
2009-09-06: cosmetic changes, rearrangements
"""
import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal

def movorder(x, order='med', windsize=3, lag='lagged'):
    """moving order statistics

    Parameters
    ----------
    x : ndarray
       time series data
    order : float or 'med', 'min', 'max'
       which order statistic to calculate
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    filtered array


    """
    pass

def check_movorder():
    """graphical test for movorder"""
    pass

def movmean(x, windowsize=3, lag='lagged'):
    """moving window mean


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving mean, with same shape as x


    Notes
    -----
    for leading and lagging the data array x is extended by the closest value of the array


    """
    pass

def movvar(x, windowsize=3, lag='lagged'):
    """moving window variance


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving variance, with same shape as x


    """
    pass

def movmoment(x, k, windowsize=3, lag='lagged'):
    """non-central moment


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        k-th moving non-central moment, with same shape as x


    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.

    """
    pass
__all__ = ['movorder', 'movmean', 'movvar', 'movmoment']
if __name__ == '__main__':
    print('\ncheckin moving mean and variance')
    nobs = 10
    x = np.arange(nobs)
    ws = 3
    ave = np.array([0.0, 1 / 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 26 / 3.0, 9])
    va = np.array([[0.0, 0.0], [0.22222222, 0.88888889], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.66666667, 2.66666667], [0.22222222, 0.88888889], [0.0, 0.0]])
    ave2d = np.c_[ave, 2 * ave]
    print(movmean(x, windowsize=ws, lag='lagged'))
    print(movvar(x, windowsize=ws, lag='lagged'))
    print([np.var(x[i - ws:i]) for i in range(ws, nobs)])
    m1 = movmoment(x, 1, windowsize=3, lag='lagged')
    m2 = movmoment(x, 2, windowsize=3, lag='lagged')
    print(m1)
    print(m2)
    print(m2 - m1 * m1)
    assert_array_almost_equal(va[ws - 1:, 0], movvar(x, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws // 2:-ws // 2 + 1, 0], movvar(x, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws + 1, 0], movvar(x, windowsize=ws, lag='lagged'))
    print('\nchecking moving moment for 2d (columns only)')
    x2d = np.c_[x, 2 * x]
    print(movmoment(x2d, 1, windowsize=3, lag='centered'))
    print(movmean(x2d, windowsize=ws, lag='lagged'))
    print(movvar(x2d, windowsize=ws, lag='lagged'))
    assert_array_almost_equal(va[ws - 1:, :], movvar(x2d, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws // 2:-ws // 2 + 1, :], movvar(x2d, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws + 1, :], movvar(x2d, windowsize=ws, lag='lagged'))
    assert_array_almost_equal(ave2d[ws - 1:], movmoment(x2d, 1, windowsize=3, lag='leading'))
    assert_array_almost_equal(ave2d[ws // 2:-ws // 2 + 1], movmoment(x2d, 1, windowsize=3, lag='centered'))
    assert_array_almost_equal(ave2d[:-ws + 1], movmean(x2d, windowsize=ws, lag='lagged'))
    from scipy import ndimage
    print(ndimage.filters.correlate1d(x2d, np.array([1, 1, 1]) / 3.0, axis=0))
    xg = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5])
    assert_array_almost_equal(xg, movmean(np.arange(100), 10, 'lagged'))
    xd = np.array([0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.4, 96.2, 96.9, 97.5, 98.0, 98.4, 98.7, 98.9, 99.0])
    assert_array_almost_equal(xd, movmean(np.arange(100), 10, 'leading'))
    xc = np.array([1.36363636, 1.90909091, 2.54545455, 3.27272727, 4.09090909, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 94.90909091, 95.72727273, 96.45454545, 97.09090909, 97.63636364])
    assert_array_almost_equal(xc, movmean(np.arange(100), 11, 'centered'))