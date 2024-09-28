from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, _calc_nodewise_weight, _calc_approx_inv_cov
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
'\nDistributed estimation routines. Currently, we support several\nmethods of distribution\n\n- sequential, has no extra dependencies\n- parallel\n    - with joblib\n        A variety of backends are supported through joblib\n        This allows for different types of clusters besides\n        standard local clusters.  Some examples of\n        backends supported by joblib are\n          - dask.distributed\n          - yarn\n          - ipyparallel\n\nThe framework is very general and allows for a variety of\nestimation methods.  Currently, these include\n\n- debiased regularized estimation\n- simple coefficient averaging (naive)\n    - regularized\n    - unregularized\n\nCurrently, the default is regularized estimation with debiasing\nwhich follows the methods outlined in\n\nJason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.\n"Communication-Efficient Sparse Regression: A One-Shot Approach."\narXiv:1503.04337. 2015. https://arxiv.org/abs/1503.04337.\n\nThere are several variables that are taken from the source paper\nfor which the interpretation may not be directly clear from the\ncode, these are mostly used to help form the estimate of the\napproximate inverse covariance matrix as part of the\ndebiasing procedure.\n\n    wexog\n\n    A weighted design matrix used to perform the node-wise\n    regression procedure.\n\n    nodewise_row\n\n    nodewise_row is produced as part of the node-wise regression\n    procedure used to produce the approximate inverse covariance\n    matrix.  One is produced for each variable using the\n    LASSO.\n\n    nodewise_weight\n\n    nodewise_weight is produced using the gamma_hat values for\n    each p to produce weights to reweight the gamma_hat values which\n    are ultimately used to form approx_inv_cov.\n\n    approx_inv_cov\n\n    This is the estimate of the approximate inverse covariance\n    matrix.  This is used to debiase the coefficient average\n    along with the average gradient.  For the OLS case,\n    approx_inv_cov is an approximation for\n\n        n * (X^T X)^{-1}\n\n    formed by node-wise regression.\n'

def _est_regularized_naive(mod, pnum, partitions, fit_kwds=None):
    """estimates the regularized fitted parameters.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    pnum : scalar
        Index of current partition
    partitions : scalar
        Total number of partitions
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized

    Returns
    -------
    An array of the parameters for the regularized fit
    """
    pass

def _est_unregularized_naive(mod, pnum, partitions, fit_kwds=None):
    """estimates the unregularized fitted parameters.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    pnum : scalar
        Index of current partition
    partitions : scalar
        Total number of partitions
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit

    Returns
    -------
    An array of the parameters for the fit
    """
    pass

def _join_naive(params_l, threshold=0):
    """joins the results from each run of _est_<type>_naive
    and returns the mean estimate of the coefficients

    Parameters
    ----------
    params_l : list
        A list of arrays of coefficients.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """
    pass

def _calc_grad(mod, params, alpha, L1_wt, score_kwds):
    """calculates the log-likelihood gradient for the debiasing

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    score_kwds : dict-like or None
        Keyword arguments for the score function.

    Returns
    -------
    An array-like object of the same dimension as params

    Notes
    -----
    In general:

    gradient l_k(params)

    where k corresponds to the index of the partition

    For OLS:

    X^T(y - X^T params)
    """
    pass

def _calc_wdesign_mat(mod, params, hess_kwds):
    """calculates the weighted design matrix necessary to generate
    the approximate inverse covariance matrix

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    hess_kwds : dict-like or None
        Keyword arguments for the hessian function.

    Returns
    -------
    An array-like object, updated design matrix, same dimension
    as mod.exog
    """
    pass

def _est_regularized_debiased(mod, mnum, partitions, fit_kwds=None, score_kwds=None, hess_kwds=None):
    """estimates the regularized fitted parameters, is the default
    estimation_method for class DistributedModel.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    mnum : scalar
        Index of current partition.
    partitions : scalar
        Total number of partitions.
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A tuple of parameters for regularized fit
        An array-like object of the fitted parameters, params
        An array-like object for the gradient
        A list of array like objects for nodewise_row
        A list of array like objects for nodewise_weight
    """
    pass

def _join_debiased(results_l, threshold=0):
    """joins the results from each run of _est_regularized_debiased
    and returns the debiased estimate of the coefficients

    Parameters
    ----------
    results_l : list
        A list of tuples each one containing the params, grad,
        nodewise_row and nodewise_weight values for each partition.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """
    pass

def _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e={}):
    """handles the model fitting for each machine. NOTE: this
    is primarily handled outside of DistributedModel because
    joblib cannot handle class methods.

    Parameters
    ----------
    self : DistributedModel class instance
        An instance of DistributedModel.
    pnum : scalar
        index of current partition.
    endog : array_like
        endogenous data for current partition.
    exog : array_like
        exogenous data for current partition.
    fit_kwds : dict-like
        Keywords needed for the model fitting.
    init_kwds_e : dict-like
        Additional init_kwds to add for each partition.

    Returns
    -------
    estimation_method result.  For the default,
    _est_regularized_debiased, a tuple.
    """
    pass

class DistributedModel:
    __doc__ = '\n    Distributed model class\n\n    Parameters\n    ----------\n    partitions : scalar\n        The number of partitions that the data will be split into.\n    model_class : statsmodels model class\n        The model class which will be used for estimation. If None\n        this defaults to OLS.\n    init_kwds : dict-like or None\n        Keywords needed for initializing the model, in addition to\n        endog and exog.\n    init_kwds_generator : generator or None\n        Additional keyword generator that produces model init_kwds\n        that may vary based on data partition.  The current usecase\n        is for WLS and GLS\n    estimation_method : function or None\n        The method that performs the estimation for each partition.\n        If None this defaults to _est_regularized_debiased.\n    estimation_kwds : dict-like or None\n        Keywords to be passed to estimation_method.\n    join_method : function or None\n        The method used to recombine the results from each partition.\n        If None this defaults to _join_debiased.\n    join_kwds : dict-like or None\n        Keywords to be passed to join_method.\n    results_class : results class or None\n        The class of results that should be returned.  If None this\n        defaults to RegularizedResults.\n    results_kwds : dict-like or None\n        Keywords to be passed to results class.\n\n    Attributes\n    ----------\n    partitions : scalar\n        See Parameters.\n    model_class : statsmodels model class\n        See Parameters.\n    init_kwds : dict-like\n        See Parameters.\n    init_kwds_generator : generator or None\n        See Parameters.\n    estimation_method : function\n        See Parameters.\n    estimation_kwds : dict-like\n        See Parameters.\n    join_method : function\n        See Parameters.\n    join_kwds : dict-like\n        See Parameters.\n    results_class : results class\n        See Parameters.\n    results_kwds : dict-like\n        See Parameters.\n\n    Notes\n    -----\n\n    Examples\n    --------\n    '

    def __init__(self, partitions, model_class=None, init_kwds=None, estimation_method=None, estimation_kwds=None, join_method=None, join_kwds=None, results_class=None, results_kwds=None):
        self.partitions = partitions
        if model_class is None:
            self.model_class = OLS
        else:
            self.model_class = model_class
        if init_kwds is None:
            self.init_kwds = {}
        else:
            self.init_kwds = init_kwds
        if estimation_method is None:
            self.estimation_method = _est_regularized_debiased
        else:
            self.estimation_method = estimation_method
        if estimation_kwds is None:
            self.estimation_kwds = {}
        else:
            self.estimation_kwds = estimation_kwds
        if join_method is None:
            self.join_method = _join_debiased
        else:
            self.join_method = join_method
        if join_kwds is None:
            self.join_kwds = {}
        else:
            self.join_kwds = join_kwds
        if results_class is None:
            self.results_class = RegularizedResults
        else:
            self.results_class = results_class
        if results_kwds is None:
            self.results_kwds = {}
        else:
            self.results_kwds = results_kwds

    def fit(self, data_generator, fit_kwds=None, parallel_method='sequential', parallel_backend=None, init_kwds_generator=None):
        """Performs the distributed estimation using the corresponding
        DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like or None
            Keywords needed for the model fitting.
        parallel_method : str
            type of distributed estimation to be used, currently
            "sequential", "joblib" and "dask" are supported.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        pass

    def fit_sequential(self, data_generator, fit_kwds, init_kwds_generator=None):
        """Sequentially performs the distributed estimation using
        the corresponding DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        pass

    def fit_joblib(self, data_generator, fit_kwds, parallel_backend, init_kwds_generator=None):
        """Performs the distributed estimation in parallel using joblib

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        pass

class DistributedResults(LikelihoodModelResults):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        Class instance for model used for distributed data,
        this particular instance uses fake data and is really
        only to allow use of methods like predict.
    params : ndarray
        Parameter estimates from the fit model.
    """

    def __init__(self, model, params):
        super(DistributedResults, self).__init__(model, params)

    def predict(self, exog, *args, **kwargs):
        """Calls self.model.predict for the provided exog.  See
        Results.predict.

        Parameters
        ----------
        exog : array_like NOT optional
            The values for which we want to predict, unlike standard
            predict this is NOT optional since the data in self.model
            is fake.
        *args :
            Some models can take additional arguments. See the
            predict method of the model for the details.
        **kwargs :
            Some models can take additional keywords arguments. See the
            predict method of the model for the details.

        Returns
        -------
            prediction : ndarray, pandas.Series or pandas.DataFrame
            See self.model.predict
        """
        pass