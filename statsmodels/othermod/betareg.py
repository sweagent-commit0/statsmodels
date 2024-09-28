u"""
Beta regression for modeling rates and proportions.

References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.

Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""
import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults, _LLRMixin
from statsmodels.genmod import families
_init_example = "\n\n    Beta regression with default of logit-link for exog and log-link\n    for precision.\n\n    >>> mod = BetaModel(endog, exog)\n    >>> rslt = mod.fit()\n    >>> print(rslt.summary())\n\n    We can also specify a formula and a specific structure and use the\n    identity-link for precision.\n\n    >>> from sm.families.links import identity\n    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')\n    >>> mod = BetaModel.from_formula('iyield ~ C(batch, Treatment(10)) + temp',\n    ...                              dat, exog_precision=Z,\n    ...                              link_precision=identity())\n\n    In the case of proportion-data, we may think that the precision depends on\n    the number of measurements. E.g for sequence data, on the number of\n    sequence reads covering a site:\n\n    >>> Z = patsy.dmatrix('~ coverage', df)\n    >>> formula = 'methylation ~ disease + age + gender + coverage'\n    >>> mod = BetaModel.from_formula(formula, df, Z)\n    >>> rslt = mod.fit()\n\n"

class BetaModel(GenericLikelihoodModel):
    __doc__ = 'Beta Regression.\n\n    The Model is parameterized by mean and precision. Both can depend on\n    explanatory variables through link functions.\n\n    Parameters\n    ----------\n    endog : array_like\n        1d array of endogenous response variable.\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user (models specified using a formula\n        include an intercept by default). See `statsmodels.tools.add_constant`.\n    exog_precision : array_like\n        2d array of variables for the precision.\n    link : link\n        Any link in sm.families.links for mean, should have range in\n        interval [0, 1]. Default is logit-link.\n    link_precision : link\n        Any link in sm.families.links for precision, should have\n        range in positive line. Default is log-link.\n    **kwds : extra keywords\n        Keyword options that will be handled by super classes.\n        Not all general keywords will be supported in this class.\n\n    Notes\n    -----\n    Status: experimental, new in 0.13.\n    Core results are verified, but api can change and some extra results\n    specific to Beta regression are missing.\n\n    Examples\n    --------\n    {example}\n\n    See Also\n    --------\n    :ref:`links`\n\n    '.format(example=_init_example)

    def __init__(self, endog, exog, exog_precision=None, link=families.links.Logit(), link_precision=families.links.Log(), **kwds):
        etmp = np.array(endog)
        assert np.all((0 < etmp) & (etmp < 1))
        if exog_precision is None:
            extra_names = ['precision']
            exog_precision = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in (exog_precision.columns if hasattr(exog_precision, 'columns') else range(1, exog_precision.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names
        super(BetaModel, self).__init__(endog, exog, exog_precision=exog_precision, **kwds)
        self.link = link
        self.link_precision = link_precision
        self.nobs = self.endog.shape[0]
        self.k_extra = 1
        self.df_model = self.nparams - 2
        self.df_resid = self.nobs - self.nparams
        assert len(self.exog_precision) == len(self.endog)
        self.hess_type = 'oim'
        if 'exog_precision' not in self._init_keys:
            self._init_keys.extend(['exog_precision'])
        self._init_keys.extend(['link', 'link_precision'])
        self._null_drop_keys = ['exog_precision']
        del kwds['extra_params_names']
        self._check_kwargs(kwds)
        self.results_class = BetaResults
        self.results_class_wrapper = BetaResultsWrapper

    def predict(self, params, exog=None, exog_precision=None, which='mean'):
        """Predict values for mean or precision

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for precision parameter.
        which : str

            - "mean" : mean, conditional expectation E(endog | exog)
            - "precision" : predicted precision
            - "linear" : linear predictor for the mean function
            - "linear-precision" : linear predictor for the precision parameter

        Returns
        -------
        ndarray, predicted values
        """
        pass

    def _predict_precision(self, params, exog_precision=None):
        """Predict values for precision function for given exog_precision.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted precision.
        """
        pass

    def _predict_var(self, params, exog=None, exog_precision=None):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of the Beta regressionmodel.

        Parameters
        ----------
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        pass

    def _llobs(self, endog, exog, exog_precision, params):
        """
        Loglikelihood for observations with data arguments.

        Parameters
        ----------
        endog : ndarray
            1d array of endogenous variable.
        exog : ndarray
            2d array of explanatory variables.
        exog_precision : ndarray
            2d array of explanatory variables for precision.
        params : ndarray
            The parameters of the model, coefficients for linear predictors
            of the mean and of the precision function.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        pass

    def score(self, params):
        """
        Returns the score vector of the log-likelihood.

        http://www.tandfonline.com/doi/pdf/10.1080/00949650903389993

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score : ndarray
            First derivative of loglikelihood function.
        """
        pass

    def _score_check(self, params):
        """Inherited score with finite differences

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score based on numerical derivatives
        """
        pass

    def score_factor(self, params, endog=None):
        """Derivative of loglikelihood function w.r.t. linear predictors.

        This needs to be multiplied with the exog to obtain the score_obs.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.

        Notes
        -----
        The score_obs can be obtained from score_factor ``sf`` using

            - d1 = sf[:, :1] * exog
            - d2 = sf[:, 1:2] * exog_precision

        """
        pass

    def score_hessian_factor(self, params, return_hessian=False, observed=True):
        """Derivatives of loglikelihood function w.r.t. linear predictors.

        This calculates score and hessian factors at the same time, because
        there is a large overlap in calculations.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        return_hessian : bool
            If False, then only score_factors are returned
            If True, the both score and hessian factors are returned
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.
        (-jbb, -jbg, -jgg) : tuple
            A tuple with 3 hessian factors, corresponding to the upper
            triangle of the Hessian matrix.
            TODO: check why there are minus
        """
        pass

    def score_obs(self, params):
        """
        Score, first derivative of the loglikelihood for each observation.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """
        pass

    def hessian(self, params, observed=None):
        """Hessian, second derivative of loglikelihood function

        Parameters
        ----------
        params : ndarray
            Parameter at which Hessian is evaluated.
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        hessian : ndarray
            Hessian, i.e. observed information, or expected information matrix.
        """
        pass

    def hessian_factor(self, params, observed=True):
        """Derivatives of loglikelihood function w.r.t. linear predictors.
        """
        pass

    def _start_params(self, niter=2, return_intermediate=False):
        """find starting values

        Parameters
        ----------
        niter : int
            Number of iterations of WLS approximation
        return_intermediate : bool
            If False (default), then only the preliminary parameter estimate
            will be returned.
            If True, then also the two results instances of the WLS estimate
            for mean parameters and for the precision parameters will be
            returned.

        Returns
        -------
        sp : ndarray
            start parameters for the optimization
        res_m2 : results instance (optional)
            Results instance for the WLS regression of the mean function.
        res_p2 : results instance (optional)
            Results instance for the WLS regression of the precision function.

        Notes
        -----
        This calculates a few iteration of weighted least squares. This is not
        a full scoring algorithm.
        """
        pass

    def fit(self, start_params=None, maxiter=1000, disp=False, method='bfgs', **kwds):
        """
        Fit the model by maximum likelihood.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        kwds :
            Keyword arguments for the optimizer.

        Returns
        -------
        BetaResults instance.
        """
        pass

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        not verified yet

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        pass

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        pass

    def get_distribution_params(self, params, exog=None, exog_precision=None):
        """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
        pass

    def get_distribution(self, params, exog=None, exog_precision=None):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.

        Returns
        -------
        Instance of a scipy frozen distribution based on estimated
        parameters.

        See Also
        --------
        predict

        Notes
        -----
        This function delegates to the predict method to handle exog and
        exog_precision, which in turn makes any required transformations.

        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        pass

class BetaResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Beta regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    @cache_readonly
    def fittedvalues(self):
        """In-sample predicted mean, conditional expectation."""
        pass

    @cache_readonly
    def fitted_precision(self):
        """In-sample predicted precision"""
        pass

    @cache_readonly
    def resid(self):
        """Response residual"""
        pass

    @cache_readonly
    def resid_pearson(self):
        """Pearson standardize residual"""
        pass

    @cache_readonly
    def prsquared(self):
        """Cox-Snell Likelihood-Ratio pseudo-R-squared.

        1 - exp((llnull - .llf) * (2 / nobs))
        """
        pass

    def get_distribution_params(self, exog=None, exog_precision=None, transform=True):
        """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
        pass

    def get_distribution(self, exog=None, exog_precision=None, transform=True):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        Instance of a scipy frozen distribution based on estimated
        parameters.

        See Also
        --------
        predict

        Notes
        -----
        This function delegates to the predict method to handle exog and
        exog_precision, which in turn makes any required transformations.

        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        pass

    def get_influence(self):
        """
        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        Support for mutli-link and multi-exog models is still experimental
        in MLEInfluence. Interface and some definitions might still change.

        Note: Difference to R betareg: Betareg has the same general leverage
        as this model. However, they use a linear approximation hat matrix
        to scale and studentize influence and residual statistics.
        MLEInfluence uses the generalized leverage as hat_matrix_diag.
        Additionally, MLEInfluence uses pearson residuals for residual
        analusis.

        References
        ----------
        todo

        """
        pass

class BetaResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(BetaResultsWrapper, BetaResults)