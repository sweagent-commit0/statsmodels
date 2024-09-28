"""Anova k-sample comparison without and with trimming

Created on Sun Jun 09 23:51:34 2013

Author: Josef Perktold
"""
import numbers
import numpy as np

def trimboth(a, proportiontocut, axis=0):
    """
    Slices off a proportion of items from both ends of an array.

    Slices off the passed proportion of items from both ends of the passed
    array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**
    rightmost 10% of scores).  You must pre-sort the array if you want
    'proper' trimming.  Slices off less if proportion results in a
    non-integer slice index (i.e., conservatively slices off
    `proportiontocut`).

    Parameters
    ----------
    a : array_like
        Data to trim.
    proportiontocut : float or int
        Proportion of data to trim at each end.
    axis : int or None
        Axis along which the observations are trimmed. The default is to trim
        along axis=0. If axis is None then the array will be flattened before
        trimming.

    Returns
    -------
    out : array-like
        Trimmed version of array `a`.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(20)
    >>> b = stats.trimboth(a, 0.1)
    >>> b.shape
    (16,)

    """
    pass

def trim_mean(a, proportiontocut, axis=0):
    """
    Return mean of array after trimming observations from both tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. Slices off LESS if proportion results in a non-integer slice
    index (i.e., conservatively slices off `proportiontocut` ).

    Parameters
    ----------
    a : array_like
        Input array
    proportiontocut : float
        Fraction to cut off at each tail of the sorted observations.
    axis : int or None
        Axis along which the trimmed means are computed. The default is axis=0.
        If axis is None then the trimmed mean will be computed for the
        flattened array.

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    """
    pass

class TrimmedMean:
    """
    class for trimmed and winsorized one sample statistics

    axis is None, i.e. ravelling, is not supported

    Parameters
    ----------
    data : array-like
        The data, observations to analyze.
    fraction : float in (0, 0.5)
        The fraction of observations to trim at each tail.
        The number of observations trimmed at each tail is
        ``int(fraction * nobs)``
    is_sorted : boolean
        Indicator if data is already sorted. By default the data is sorted
        along ``axis``.
    axis : int
        The axis of reduce operations. By default axis=0, that is observations
        are along the zero dimension, i.e. rows if 2-dim.
    """

    def __init__(self, data, fraction, is_sorted=False, axis=0):
        self.data = np.asarray(data)
        self.axis = axis
        self.fraction = fraction
        self.nobs = nobs = self.data.shape[axis]
        self.lowercut = lowercut = int(fraction * nobs)
        self.uppercut = uppercut = nobs - lowercut
        if lowercut >= uppercut:
            raise ValueError('Proportion too big.')
        self.nobs_reduced = nobs - 2 * lowercut
        self.sl = [slice(None)] * self.data.ndim
        self.sl[axis] = slice(self.lowercut, self.uppercut)
        self.sl = tuple(self.sl)
        if not is_sorted:
            self.data_sorted = np.sort(self.data, axis=axis)
        else:
            self.data_sorted = self.data
        self.lowerbound = np.take(self.data_sorted, lowercut, axis=axis)
        self.upperbound = np.take(self.data_sorted, uppercut - 1, axis=axis)

    @property
    def data_trimmed(self):
        """numpy array of trimmed and sorted data
        """
        pass

    @property
    def data_winsorized(self):
        """winsorized data
        """
        pass

    @property
    def mean_trimmed(self):
        """mean of trimmed data
        """
        pass

    @property
    def mean_winsorized(self):
        """mean of winsorized data
        """
        pass

    @property
    def var_winsorized(self):
        """variance of winsorized data
        """
        pass

    @property
    def std_mean_trimmed(self):
        """standard error of trimmed mean
        """
        pass

    @property
    def std_mean_winsorized(self):
        """standard error of winsorized mean
        """
        pass

    def ttest_mean(self, value=0, transform='trimmed', alternative='two-sided'):
        """
        One sample t-test for trimmed or Winsorized mean

        Parameters
        ----------
        value : float
            Value of the mean under the Null hypothesis
        transform : {'trimmed', 'winsorized'}
            Specified whether the mean test is based on trimmed or winsorized
            data.
        alternative : {'two-sided', 'larger', 'smaller'}


        Notes
        -----
        p-value is based on the approximate t-distribution of the test
        statistic. The approximation is valid if the underlying distribution
        is symmetric.
        """
        pass

    def reset_fraction(self, frac):
        """create a TrimmedMean instance with a new trimming fraction

        This reuses the sorted array from the current instance.
        """
        pass

def scale_transform(data, center='median', transform='abs', trim_frac=0.2, axis=0):
    """Transform data for variance comparison for Levene type tests

    Parameters
    ----------
    data : array_like
        Observations for the data.
    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : 'abs', 'square', 'identity' or a callable
        The transform for the centered data.
    trim_frac : float in [0, 0.5)
        Fraction of observations that are trimmed on each side of the sorted
        observations. This is only used if center is `trimmed`.
    axis : int
        Axis along which the data are transformed when centering.

    Returns
    -------
    res : ndarray
        transformed data in the same shape as the original data.

    """
    pass