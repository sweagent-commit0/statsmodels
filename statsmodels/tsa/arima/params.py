"""
SARIMAX parameters class.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic

class SARIMAXParams:
    """
    SARIMAX parameters.

    Parameters
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.

    Attributes
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.
    exog_names : list of str
        Names associated with exogenous parameters.
    ar_names : list of str
        Names associated with (non-seasonal) autoregressive parameters.
    ma_names : list of str
        Names associated with (non-seasonal) moving average parameters.
    seasonal_ar_names : list of str
        Names associated with seasonal autoregressive parameters.
    seasonal_ma_names : list of str
        Names associated with seasonal moving average parameters.
    param_names :list of str
        Names of all model parameters.
    k_exog_params : int
        Number of parameters associated with exogenous variables.
    k_ar_params : int
        Number of parameters associated with (non-seasonal) autoregressive
        lags.
    k_ma_params : int
        Number of parameters associated with (non-seasonal) moving average
        lags.
    k_seasonal_ar_params : int
        Number of parameters associated with seasonal autoregressive lags.
    k_seasonal_ma_params : int
        Number of parameters associated with seasonal moving average lags.
    k_params : int
        Total number of model parameters.
    """

    def __init__(self, spec):
        self.spec = spec
        self.exog_names = spec.exog_names
        self.ar_names = spec.ar_names
        self.ma_names = spec.ma_names
        self.seasonal_ar_names = spec.seasonal_ar_names
        self.seasonal_ma_names = spec.seasonal_ma_names
        self.param_names = spec.param_names
        self.k_exog_params = spec.k_exog_params
        self.k_ar_params = spec.k_ar_params
        self.k_ma_params = spec.k_ma_params
        self.k_seasonal_ar_params = spec.k_seasonal_ar_params
        self.k_seasonal_ma_params = spec.k_seasonal_ma_params
        self.k_params = spec.k_params
        self._params_split = spec.split_params(np.zeros(self.k_params) * np.nan, allow_infnan=True)
        self._params = None

    @property
    def exog_params(self):
        """(array) Parameters associated with exogenous variables."""
        pass

    @property
    def ar_params(self):
        """(array) Autoregressive (non-seasonal) parameters."""
        pass

    @property
    def ar_poly(self):
        """(Polynomial) Autoregressive (non-seasonal) lag polynomial."""
        pass

    @property
    def ma_params(self):
        """(array) Moving average (non-seasonal) parameters."""
        pass

    @property
    def ma_poly(self):
        """(Polynomial) Moving average (non-seasonal) lag polynomial."""
        pass

    @property
    def seasonal_ar_params(self):
        """(array) Seasonal autoregressive parameters."""
        pass

    @property
    def seasonal_ar_poly(self):
        """(Polynomial) Seasonal autoregressive lag polynomial."""
        pass

    @property
    def seasonal_ma_params(self):
        """(array) Seasonal moving average parameters."""
        pass

    @property
    def seasonal_ma_poly(self):
        """(Polynomial) Seasonal moving average lag polynomial."""
        pass

    @property
    def sigma2(self):
        """(float) Innovation variance."""
        pass

    @property
    def reduced_ar_poly(self):
        """(Polynomial) Reduced form autoregressive lag polynomial."""
        pass

    @property
    def reduced_ma_poly(self):
        """(Polynomial) Reduced form moving average lag polynomial."""
        pass

    @property
    def params(self):
        """(array) Complete parameter vector."""
        pass

    @property
    def is_complete(self):
        """(bool) Are current parameter values all filled in (i.e. not NaN)."""
        pass

    @property
    def is_valid(self):
        """(bool) Are current parameter values valid (e.g. variance > 0)."""
        pass

    @property
    def is_stationary(self):
        """(bool) Is the reduced autoregressive lag poylnomial stationary."""
        pass

    @property
    def is_invertible(self):
        """(bool) Is the reduced moving average lag poylnomial invertible."""
        pass

    def to_dict(self):
        """
        Return the parameters split by type into a dictionary.

        Returns
        -------
        split_params : dict
            Dictionary with keys 'exog_params', 'ar_params', 'ma_params',
            'seasonal_ar_params', 'seasonal_ma_params', and (unless
            `concentrate_scale=True`) 'sigma2'. Values are the parameters
            associated with the key, based on the `params` argument.
        """
        pass

    def to_pandas(self):
        """
        Return the parameters as a Pandas series.

        Returns
        -------
        series : pd.Series
            Pandas series with index set to the parameter names.
        """
        pass

    def __repr__(self):
        """Represent SARIMAXParams object as a string."""
        components = []
        if self.k_exog_params:
            components.append('exog=%s' % str(self.exog_params))
        if self.k_ar_params:
            components.append('ar=%s' % str(self.ar_params))
        if self.k_ma_params:
            components.append('ma=%s' % str(self.ma_params))
        if self.k_seasonal_ar_params:
            components.append('seasonal_ar=%s' % str(self.seasonal_ar_params))
        if self.k_seasonal_ma_params:
            components.append('seasonal_ma=%s' % str(self.seasonal_ma_params))
        if not self.spec.concentrate_scale:
            components.append('sigma2=%s' % self.sigma2)
        return 'SARIMAXParams(%s)' % ', '.join(components)