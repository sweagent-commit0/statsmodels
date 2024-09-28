import numpy as np
from statsmodels.duration.hazard_regression import PHReg

def _kernel_cumincidence(time, status, exog, kfunc, freq_weights, dimred=True):
    """
    Calculates cumulative incidence functions using kernels.

    Parameters
    ----------
    time : array_like
        The observed time values
    status : array_like
        The status values.  status == 0 indicates censoring,
        status == 1, 2, ... are the events.
    exog : array_like
        Covariates such that censoring becomes independent of
        outcome times conditioned on the covariate values.
    kfunc : function
        A kernel function
    freq_weights : array_like
        Optional frequency weights
    dimred : bool
        If True, proportional hazards regression models are used to
        reduce exog to two columns by predicting overall events and
        censoring in two separate models.  If False, exog is used
        directly for calculating kernel weights without dimension
        reduction.
    """
    pass

def _kernel_survfunc(time, status, exog, kfunc, freq_weights):
    """
    Estimate the marginal survival function under dependent censoring.

    Parameters
    ----------
    time : array_like
        The observed times for each subject
    status : array_like
        The status for each subject (1 indicates event, 0 indicates
        censoring)
    exog : array_like
        Covariates such that censoring is independent conditional on
        exog
    kfunc : function
        Kernel function
    freq_weights : array_like
        Optional frequency weights

    Returns
    -------
    probs : array_like
        The estimated survival probabilities
    times : array_like
        The times at which the survival probabilities are estimated

    References
    ----------
    Zeng, Donglin 2004. Estimating Marginal Survival Function by
    Adjusting for Dependent Censoring Using Many Covariates. The
    Annals of Statistics 32 (4): 1533 55.
    doi:10.1214/009053604000000508.
    https://arxiv.org/pdf/math/0409180.pdf
    """
    pass