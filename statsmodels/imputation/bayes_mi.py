import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults

class BayesGaussMI:
    """
    Bayesian Imputation using a Gaussian model.

    The approach is Bayesian.  The goal is to sample from the joint
    distribution of the mean vector, covariance matrix, and missing
    data values given the observed data values.  Conjugate priors for
    the population mean and covariance matrix are used.  Gibbs
    sampling is used to update the mean vector, covariance matrix, and
    missing data values in turn.  After burn-in, the imputed complete
    data sets from the Gibbs chain can be used in multiple imputation
    analyses (MI).

    Parameters
    ----------
    data : ndarray
        The array of data to be imputed.  Values in the array equal to
        NaN are imputed.
    mean_prior : ndarray, optional
        The covariance matrix of the Gaussian prior distribution for
        the mean vector.  If not provided, the identity matrix is
        used.
    cov_prior : ndarray, optional
        The center matrix for the inverse Wishart prior distribution
        for the covariance matrix.  If not provided, the identity
        matrix is used.
    cov_prior_df : positive float
        The degrees of freedom of the inverse Wishart prior
        distribution for the covariance matrix.  Defaults to 1.

    Examples
    --------
    A basic example with OLS. Data is generated assuming 10% is missing at
    random.

    >>> import numpy as np
    >>> x = np.random.standard_normal((1000, 2))
    >>> x.flat[np.random.sample(2000) < 0.1] = np.nan

    The imputer is used with ``MI``.

    >>> import statsmodels.api as sm
    >>> def model_args_fn(x):
    ...     # Return endog, exog from x
    ...    return x[:, 0], x[:, 1:]
    >>> imp = sm.BayesGaussMI(x)
    >>> mi = sm.MI(imp, sm.OLS, model_args_fn)
    """

    def __init__(self, data, mean_prior=None, cov_prior=None, cov_prior_df=1):
        self.exog_names = None
        if type(data) is pd.DataFrame:
            self.exog_names = data.columns
        data = np.require(data, requirements='W')
        self.data = data
        self._data = data
        self.mask = np.isnan(data)
        self.nobs = self.mask.shape[0]
        self.nvar = self.mask.shape[1]
        z = 1 + np.log(1 + np.arange(self.mask.shape[1]))
        c = np.dot(self.mask, z)
        rowmap = {}
        for i, v in enumerate(c):
            if v == 0:
                continue
            if v not in rowmap:
                rowmap[v] = []
            rowmap[v].append(i)
        self.patterns = [np.asarray(v) for v in rowmap.values()]
        p = self._data.shape[1]
        self.cov = np.eye(p)
        mean = []
        for i in range(p):
            v = self._data[:, i]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                msg = 'Column %d has no observed values' % i
                raise ValueError(msg)
            mean.append(v.mean())
        self.mean = np.asarray(mean)
        if mean_prior is None:
            mean_prior = np.eye(p)
        self.mean_prior = mean_prior
        if cov_prior is None:
            cov_prior = np.eye(p)
        self.cov_prior = cov_prior
        self.cov_prior_df = cov_prior_df

    def update(self):
        """
        Cycle through all Gibbs updates.
        """
        pass

    def update_data(self):
        """
        Gibbs update of the missing data values.
        """
        pass

    def update_mean(self):
        """
        Gibbs update of the mean vector.

        Do not call until update_data has been called once.
        """
        pass

    def update_cov(self):
        """
        Gibbs update of the covariance matrix.

        Do not call until update_data has been called once.
        """
        pass

class MI:
    """
    MI performs multiple imputation using a provided imputer object.

    Parameters
    ----------
    imp : object
        An imputer class, such as BayesGaussMI.
    model : model class
        Any statsmodels model class.
    model_args_fn : function
        A function taking an imputed dataset as input and returning
        endog, exog.  If the model is fit using a formula, returns
        a DataFrame used to build the model.  Optional when a formula
        is used.
    model_kwds_fn : function, optional
        A function taking an imputed dataset as input and returning
        a dictionary of model keyword arguments.
    formula : str, optional
        If provided, the model is constructed using the `from_formula`
        class method, otherwise the `__init__` method is used.
    fit_args : list-like, optional
        List of arguments to be passed to the fit method
    fit_kwds : dict-like, optional
        Keyword arguments to be passed to the fit method
    xfunc : function mapping ndarray to ndarray
        A function that is applied to the complete data matrix
        prior to fitting the model
    burn : int
        Number of burn-in iterations
    nrep : int
        Number of imputed data sets to use in the analysis
    skip : int
        Number of Gibbs iterations to skip between successive
        multiple imputation fits.

    Notes
    -----
    The imputer object must have an 'update' method, and a 'data'
    attribute that contains the current imputed dataset.

    xfunc can be used to introduce domain constraints, e.g. when
    imputing binary data the imputed continuous values can be rounded
    to 0/1.
    """

    def __init__(self, imp, model, model_args_fn=None, model_kwds_fn=None, formula=None, fit_args=None, fit_kwds=None, xfunc=None, burn=100, nrep=20, skip=10):
        self.imp = imp
        self.skip = skip
        self.model = model
        self.formula = formula
        if model_args_fn is None:

            def f(x):
                return []
            model_args_fn = f
        self.model_args_fn = model_args_fn
        if model_kwds_fn is None:

            def f(x):
                return {}
            model_kwds_fn = f
        self.model_kwds_fn = model_kwds_fn
        if fit_args is None:

            def f(x):
                return []
            fit_args = f
        self.fit_args = fit_args
        if fit_kwds is None:

            def f(x):
                return {}
            fit_kwds = f
        self.fit_kwds = fit_kwds
        self.xfunc = xfunc
        self.nrep = nrep
        self.skip = skip
        for k in range(burn):
            imp.update()

    def fit(self, results_cb=None):
        """
        Impute datasets, fit models, and pool results.

        Parameters
        ----------
        results_cb : function, optional
            If provided, each results instance r is passed through `results_cb`,
            then appended to the `results` attribute of the MIResults object.
            To save complete results, use `results_cb=lambda x: x`.  The default
            behavior is to save no results.

        Returns
        -------
        A MIResults object.
        """
        pass

class MIResults(LikelihoodModelResults):
    """
    A results class for multiple imputation (MI).

    Parameters
    ----------
    mi : MI instance
        The MI object that produced the results
    model : instance of statsmodels model class
        This can be any instance from the multiple imputation runs.
        It is used to get class information, the specific parameter
        and data values are not used.
    params : array_like
        The overall multiple imputation parameter estimates.
    normalized_cov_params : array_like (2d)
        The overall variance covariance matrix of the estimates.
    """

    def __init__(self, mi, model, params, normalized_cov_params):
        super(MIResults, self).__init__(model, params, normalized_cov_params)
        self.mi = mi
        self._model = model

    def summary(self, title=None, alpha=0.05):
        """
        Summarize the results of running multiple imputation.

        Parameters
        ----------
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        """
        pass