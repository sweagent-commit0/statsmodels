"""
Hannan-Rissanen procedure for estimating ARMA(p,q) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

def hannan_rissanen(endog, ar_order=0, ma_order=0, demean=True, initial_ar_order=None, unbiased=None, fixed_params=None):
    """
    Estimate ARMA parameters using Hannan-Rissanen procedure.

    Parameters
    ----------
    endog : array_like
        Input time series array, assumed to be stationary.
    ar_order : int or list of int
        Autoregressive order
    ma_order : int or list of int
        Moving average order
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the ARMA coefficients. Default is True.
    initial_ar_order : int, optional
        Order of long autoregressive process used for initial computation of
        residuals.
    unbiased : bool, optional
        Whether or not to apply the bias correction step. Default is True if
        the estimated coefficients from the previous step imply a stationary
        and invertible process and False otherwise.
    fixed_params : dict, optional
        Dictionary with names of fixed parameters as keys (e.g. 'ar.L1',
        'ma.L2'), which correspond to SARIMAXSpecification.param_names.
        Dictionary values are the values of the associated fixed parameters.

    Returns
    -------
    parameters : SARIMAXParams object
    other_results : Bunch
        Includes three components: `spec`, containing the
        `SARIMAXSpecification` instance corresponding to the input arguments;
        `initial_ar_order`, containing the autoregressive lag order used in the
        first step; and `resid`, which contains the computed residuals from the
        last step.

    Notes
    -----
    The primary reference is [1]_, section 5.1.4, which describes a three-step
    procedure that we implement here.

    1. Fit a large-order AR model via Yule-Walker to estimate residuals
    2. Compute AR and MA estimates via least squares
    3. (Unless the estimated coefficients from step (2) are non-stationary /
       non-invertible or `unbiased=False`) Perform bias correction

    The order used for the AR model in the first step may be given as an
    argument. If it is not, we compute it as suggested by [2]_.

    The estimate of the variance that we use is computed from the residuals
    of the least-squares regression and not from the innovations algorithm.
    This is because our fast implementation of the innovations algorithm is
    only valid for stationary processes, and the Hannan-Rissanen procedure may
    produce estimates that imply non-stationary processes. To avoid
    inconsistency, we never compute this latter variance here, even if it is
    possible. See test_hannan_rissanen::test_brockwell_davis_example_517 for
    an example of how to compute this variance manually.

    This procedure assumes that the series is stationary, but if this is not
    true, it is still possible that this procedure will return parameters that
    imply a non-stationary / non-invertible process.

    Note that the third stage will only be applied if the parameters from the
    second stage imply a stationary / invertible model. If `unbiased=True` is
    given, then non-stationary / non-invertible parameters in the second stage
    will throw an exception.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    .. [2] Gomez, Victor, and Agustin Maravall. 2001.
       "Automatic Modeling Methods for Univariate Series."
       A Course in Time Series Analysis, 171–201.
    """
    pass

def _validate_fixed_params(fixed_params, spec_param_names):
    """
    Check that keys in fixed_params are a subset of spec.param_names except
    "sigma2"

    Parameters
    ----------
    fixed_params : dict
    spec_param_names : list of string
        SARIMAXSpecification.param_names
    """
    pass

def _package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags):
    """
    Parameters
    ----------
    fixed_params : dict
    spec_ar_lags : list of int
        SARIMAXSpecification.ar_lags
    spec_ma_lags : list of int
        SARIMAXSpecification.ma_lags

    Returns
    -------
    Bunch with
    (lags) fixed_ar_lags, fixed_ma_lags, free_ar_lags, free_ma_lags;
    (ix) fixed_ar_ix, fixed_ma_ix, free_ar_ix, free_ma_ix;
    (params) fixed_ar_params, free_ma_params
    """
    pass

def _stitch_fixed_and_free_params(fixed_ar_or_ma_lags, fixed_ar_or_ma_params, free_ar_or_ma_lags, free_ar_or_ma_params, spec_ar_or_ma_lags):
    """
    Stitch together fixed and free params, by the order of lags, for setting
    SARIMAXParams.ma_params or SARIMAXParams.ar_params

    Parameters
    ----------
    fixed_ar_or_ma_lags : list or np.array
    fixed_ar_or_ma_params : list or np.array
        fixed_ar_or_ma_params corresponds with fixed_ar_or_ma_lags
    free_ar_or_ma_lags : list or np.array
    free_ar_or_ma_params : list or np.array
        free_ar_or_ma_params corresponds with free_ar_or_ma_lags
    spec_ar_or_ma_lags : list
        SARIMAXSpecification.ar_lags or SARIMAXSpecification.ma_lags

    Returns
    -------
    list of fixed and free params by the order of lags
    """
    pass