"""
Seasonal Decomposition by Moving Averages
"""
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
__all__ = ['STL', 'seasonal_decompose', 'seasonal_mean', 'DecomposeResult', 'MSTL']

def _extrapolate_trend(trend, npoints):
    """
    Replace nan values on trend's end-points with least-squares extrapolated
    values with regression considering npoints closest defined points.
    """
    pass

def seasonal_mean(x, period):
    """
    Return means for each period in x. period is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    pass

def seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0):
    """
    Seasonal decomposition using moving averages.

    Parameters
    ----------
    x : array_like
        Time series. If 2d, individual series are in columns. x must contain 2
        complete cycles.
    model : {"additive", "multiplicative"}, optional
        Type of seasonal component. Abbreviations are accepted.
    filt : array_like, optional
        The filter coefficients for filtering out the seasonal component.
        The concrete moving average method used in filtering is determined by
        two_sided.
    period : int, optional
        Period of the series. Must be used if x is not a pandas object or if
        the index of x does not have  a frequency. Overrides default
        periodicity of x if x is a pandas object with a timeseries index.
    two_sided : bool, optional
        The moving average method used in filtering.
        If True (default), a centered moving average is computed using the
        filt. If False, the filter coefficients are for past values only.
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.

    Returns
    -------
    DecomposeResult
        A object with seasonal, trend, and resid attributes.

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King filter.
    statsmodels.tsa.filters.cf_filter.cffilter
        Christiano-Fitzgerald asymmetric, random walk filter.
    statsmodels.tsa.filters.hp_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.filters.convolution_filter
        Linear filtering via convolution.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    This is a naive decomposition. More sophisticated methods should
    be preferred.

    The additive model is Y[t] = T[t] + S[t] + e[t]

    The multiplicative model is Y[t] = T[t] * S[t] * e[t]

    The results are obtained by first estimating the trend by applying
    a convolution filter to the data. The trend is then removed from the
    series and the average of this de-trended series for each period is
    the returned seasonal component.
    """
    pass

class DecomposeResult:
    """
    Results class for seasonal decompositions

    Parameters
    ----------
    observed : array_like
        The data series that has been decomposed.
    seasonal : array_like
        The seasonal component of the data series.
    trend : array_like
        The trend component of the data series.
    resid : array_like
        The residual component of the data series.
    weights : array_like, optional
        The weights used to reduce outlier influence.
    """

    def __init__(self, observed, seasonal, trend, resid, weights=None):
        self._seasonal = seasonal
        self._trend = trend
        if weights is None:
            weights = np.ones_like(observed)
            if isinstance(observed, pd.Series):
                weights = pd.Series(weights, index=observed.index, name='weights')
        self._weights = weights
        self._resid = resid
        self._observed = observed

    @property
    def observed(self):
        """Observed data"""
        pass

    @property
    def seasonal(self):
        """The estimated seasonal component"""
        pass

    @property
    def trend(self):
        """The estimated trend component"""
        pass

    @property
    def resid(self):
        """The estimated residuals"""
        pass

    @property
    def weights(self):
        """The weights used in the robust estimation"""
        pass

    @property
    def nobs(self):
        """Number of observations"""
        pass

    def plot(self, observed=True, seasonal=True, trend=True, resid=True, weights=False):
        """
        Plot estimated components

        Parameters
        ----------
        observed : bool
            Include the observed series in the plot
        seasonal : bool
            Include the seasonal component in the plot
        trend : bool
            Include the trend component in the plot
        resid : bool
            Include the residual in the plot
        weights : bool
            Include the weights in the plot (if any)

        Returns
        -------
        matplotlib.figure.Figure
            The figure instance that containing the plot.
        """
        pass