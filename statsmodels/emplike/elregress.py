"""
Empirical Likelihood Linear Regression Inference

The script contains the function that is optimized over nuisance parameters to
 conduct inference on linear regression parameters.  It is called by eltest
in OLSResults.


General References
-----------------

Owen, A.B.(2001). Empirical Likelihood. Chapman and Hall

"""
import numpy as np
from statsmodels.emplike.descriptive import _OptFuncts

class _ELRegOpts(_OptFuncts):
    """

    A class that holds functions to be optimized over when conducting
    hypothesis tests and calculating confidence intervals.

    Parameters
    ----------

    OLSResults : Results instance
        A fitted OLS result.
    """

    def __init__(self):
        pass

    def _opt_nuis_regress(self, nuisance_params, param_nums=None, endog=None, exog=None, nobs=None, nvar=None, params=None, b0_vals=None, stochastic_exog=None):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest.

        Parameters
        ----------
        nuisance_params: 1darray
            Parameters to be optimized over.

        Returns
        -------
        llr : float
            -2 x the log-likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
        pass