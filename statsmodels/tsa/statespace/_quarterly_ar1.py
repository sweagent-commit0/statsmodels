"""
(Internal) AR(1) model for monthly growth rates aggregated to quarterly freq.

Author: Chad Fulton
License: BSD-3
"""
import warnings
import numpy as np
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.kalman_smoother import SMOOTHER_STATE, SMOOTHER_STATE_COV, SMOOTHER_STATE_AUTOCOV
from statsmodels.tsa.statespace.tools import constrain_stationary_univariate, unconstrain_stationary_univariate

class QuarterlyAR1(mlemodel.MLEModel):
    """
    AR(1) model for monthly growth rates aggregated to quarterly frequency

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`

    Notes
    -----
    This model is internal, used to estimate starting parameters for the
    DynamicFactorMQ class. The model is:

    .. math::

        y_t & = \\begin{bmatrix} 1 & 2 & 3 & 2 & 1 \\end{bmatrix} \\alpha_t \\\\
        \\alpha_t & = \\begin{bmatrix}
            \\phi & 0 & 0 & 0 & 0 \\\\
               1 & 0 & 0 & 0 & 0 \\\\
               0 & 1 & 0 & 0 & 0 \\\\
               0 & 0 & 1 & 0 & 0 \\\\
               0 & 0 & 0 & 1 & 0 \\\\
        \\end{bmatrix} +
        \\begin{bmatrix} 1 \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0 \\end{bmatrix} \\varepsilon_t

    The two parameters to be estimated are :math:`\\phi` and :math:`\\sigma^2`.

    It supports fitting via the usual quasi-Newton methods, as well as using
    the EM algorithm.

    """

    def __init__(self, endog):
        super().__init__(endog, k_states=5, k_posdef=1, initialization='stationary')
        self['design'] = [1, 2, 3, 2, 1]
        self['transition', 1:, :-1] = np.eye(4)
        self['selection', 0, 0] = 1.0