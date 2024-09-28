"""
Implementation of proportional hazards regression models for duration
data that may be censored ("Cox models").

References
----------
T Therneau (1996).  Extending the Cox model.  Technical report.
http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

G Rodriguez (2005).  Non-parametric estimation in survival models.
http://data.princeton.edu/pop509/NonParametricSurvival.pdf

B Gillespie (2006).  Checking the assumptions in the Cox proportional
hazards model.
http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf
"""
import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
_predict_docstring = "\n    Returns predicted values from the proportional hazards\n    regression model.\n\n    Parameters\n    ----------%(params_doc)s\n    exog : array_like\n        Data to use as `exog` in forming predictions.  If not\n        provided, the `exog` values from the model used to fit the\n        data are used.%(cov_params_doc)s\n    endog : array_like\n        Duration (time) values at which the predictions are made.\n        Only used if pred_type is either 'cumhaz' or 'surv'.  If\n        using model `exog`, defaults to model `endog` (time), but\n        may be provided explicitly to make predictions at\n        alternative times.\n    strata : array_like\n        A vector of stratum values used to form the predictions.\n        Not used (may be 'None') if pred_type is 'lhr' or 'hr'.\n        If `exog` is None, the model stratum values are used.  If\n        `exog` is not None and pred_type is 'surv' or 'cumhaz',\n        stratum values must be provided (unless there is only one\n        stratum).\n    offset : array_like\n        Offset values used to create the predicted values.\n    pred_type : str\n        If 'lhr', returns log hazard ratios, if 'hr' returns\n        hazard ratios, if 'surv' returns the survival function, if\n        'cumhaz' returns the cumulative hazard function.\n    pred_only : bool\n        If True, returns only an array of predicted values.  Otherwise\n        returns a bunch containing the predicted values and standard\n        errors.\n\n    Returns\n    -------\n    A bunch containing two fields: `predicted_values` and\n    `standard_errors`.\n\n    Notes\n    -----\n    Standard errors are only returned when predicting the log\n    hazard ratio (pred_type is 'lhr').\n\n    Types `surv` and `cumhaz` require estimation of the cumulative\n    hazard function.\n"
_predict_params_doc = '\n    params : array_like\n        The proportional hazards model parameters.'
_predict_cov_params_docstring = "\n    cov_params : array_like\n        The covariance matrix of the estimated `params` vector,\n        used to obtain prediction errors if pred_type='lhr',\n        otherwise optional."

class PHSurvivalTime:

    def __init__(self, time, status, exog, strata=None, entry=None, offset=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.

        Parameters
        ----------
        time : array_like
            The times at which either the event (failure) occurs or
            the observation is censored.
        status : array_like
            Indicates whether the event (failure) occurs at `time`
            (`status` is 1), or if `time` is a censoring time (`status`
            is 0).
        exog : array_like
            The exogeneous (covariate) data matrix, cases are rows and
            variables are columns.
        strata : array_like
            Grouping variable defining the strata.  If None, all
            observations are in a single stratum.
        entry : array_like
            Entry (left truncation) times.  The observation is not
            part of the risk set for times before the entry time.  If
            None, the entry time is treated as being zero, which
            gives no left truncation.  The entry time must be less
            than or equal to `time`.
        offset : array_like
            An optional array of offsets
        """
        if strata is None:
            strata = np.zeros(len(time), dtype=np.int32)
        if entry is None:
            entry = np.zeros(len(time))
        self._check(time, status, strata, entry)
        stu = np.unique(strata)
        sth = {x: [] for x in stu}
        for i, k in enumerate(strata):
            sth[k].append(i)
        stratum_rows = [np.asarray(sth[k], dtype=np.int32) for k in stu]
        stratum_names = stu
        ix = [i for i, ix in enumerate(stratum_rows) if status[ix].sum() > 0]
        self.nstrat_orig = len(stratum_rows)
        stratum_rows = [stratum_rows[i] for i in ix]
        stratum_names = [stratum_names[i] for i in ix]
        nstrat = len(stratum_rows)
        self.nstrat = nstrat
        for stx, ix in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])
            ii = [i for i, t in enumerate(entry[ix]) if t <= last_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for stx, ix in enumerate(stratum_rows):
            first_failure = min(time[ix][status[ix] == 1])
            ii = [i for i, t in enumerate(time[ix]) if t >= first_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for stx, ix in enumerate(stratum_rows):
            ii = np.argsort(time[ix])
            stratum_rows[stx] = stratum_rows[stx][ii]
        if offset is not None:
            self.offset_s = []
            for stx in range(nstrat):
                self.offset_s.append(offset[stratum_rows[stx]])
        else:
            self.offset_s = None
        self.n_obs = sum([len(ix) for ix in stratum_rows])
        self.stratum_rows = stratum_rows
        self.stratum_names = stratum_names
        self.time_s = self._split(time)
        self.exog_s = self._split(exog)
        self.status_s = self._split(status)
        self.entry_s = self._split(entry)
        self.ufailt_ix, self.risk_enter, self.risk_exit, self.ufailt = ([], [], [], [])
        for stx in range(self.nstrat):
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]
            uft = np.unique(ft)
            nuft = len(uft)
            uft_map = dict([(x, i) for i, x in enumerate(uft)])
            uft_ix = [[] for k in range(nuft)]
            for ix, ti in zip(ift, ft):
                uft_ix[uft_map[ti]].append(ix)
            risk_enter1 = [[] for k in range(nuft)]
            for i, t in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, 'right') - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)
            risk_exit1 = [[] for k in range(nuft)]
            for i, t in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)
            self.ufailt.append(uft)
            self.ufailt_ix.append([np.asarray(x, dtype=np.int32) for x in uft_ix])
            self.risk_enter.append([np.asarray(x, dtype=np.int32) for x in risk_enter1])
            self.risk_exit.append([np.asarray(x, dtype=np.int32) for x in risk_exit1])

class PHReg(model.LikelihoodModel):
    """
    Cox Proportional Hazards Regression Model

    The Cox PH Model is for right censored data.

    Parameters
    ----------
    endog : array_like
        The observed times (event or censoring)
    exog : 2D array_like
        The covariates or exogeneous variables
    status : array_like
        The censoring status values; status=1 indicates that an
        event occurred (e.g. failure or death), status=0 indicates
        that the observation was right censored. If None, defaults
        to status=1 for all cases.
    entry : array_like
        The entry times, if left truncation occurs
    strata : array_like
        Stratum labels.  If None, all observations are taken to be
        in a single stratum.
    ties : str
        The method used to handle tied times, must be either 'breslow'
        or 'efron'.
    offset : array_like
        Array of offset values
    missing : str
        The method used to handle missing data

    Notes
    -----
    Proportional hazards regression models should not include an
    explicit or implicit intercept.  The effect of an intercept is
    not identified using the partial likelihood approach.

    `endog`, `event`, `strata`, `entry`, and the first dimension
    of `exog` all must have the same length
    """

    def __init__(self, endog, exog, status=None, entry=None, strata=None, offset=None, ties='breslow', missing='drop', **kwargs):
        if status is None:
            status = np.ones(len(endog))
        super(PHReg, self).__init__(endog, exog, status=status, entry=entry, strata=strata, offset=offset, missing=missing, **kwargs)
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)
        if self.offset is not None:
            self.offset = np.asarray(self.offset)
        self.surv = PHSurvivalTime(self.endog, self.status, self.exog, self.strata, self.entry, self.offset)
        self.nobs = len(self.endog)
        self.groups = None
        self.missing = missing
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog))
        ties = ties.lower()
        if ties not in ('efron', 'breslow'):
            raise ValueError('`ties` must be either `efron` or ' + '`breslow`')
        self.ties = ties

    @classmethod
    def from_formula(cls, formula, data, status=None, entry=None, strata=None, offset=None, subset=None, ties='breslow', missing='drop', *args, **kwargs):
        """
        Create a proportional hazards regression model from a formula
        and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array_like
            The data for the model. See Notes.
        status : array_like
            The censoring status values; status=1 indicates that an
            event occurred (e.g. failure or death), status=0 indicates
            that the observation was right censored. If None, defaults
            to status=1 for all cases.
        entry : array_like
            The entry times, if left truncation occurs
        strata : array_like
            Stratum labels.  If None, all observations are taken to be
            in a single stratum.
        offset : array_like
            Array of offset values
        subset : array_like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        ties : str
            The method used to handle tied times, must be either 'breslow'
            or 'efron'.
        missing : str
            The method used to handle missing data
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : PHReg model instance
        """
        pass

    def fit(self, groups=None, **args):
        """
        Fit a proportional hazards regression model.

        Parameters
        ----------
        groups : array_like
            Labels indicating groups of observations that may be
            dependent.  If present, the standard errors account for
            this dependence. Does not affect fitted values.

        Returns
        -------
        PHRegResults
            Returns a results instance.
        """
        pass

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments used to fit the model.

        Returns
        -------
        PHRegResults
            Returns a results instance.

        Notes
        -----
        The penalty is the ``elastic net`` penalty, which is a
        combination of L1 and L2 penalties.

        The function that is minimized is:

        .. math::

            -loglike/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)

        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        """
        pass

    def loglike(self, params):
        """
        Returns the log partial likelihood function evaluated at
        `params`.
        """
        pass

    def score(self, params):
        """
        Returns the score function evaluated at `params`.
        """
        pass

    def hessian(self, params):
        """
        Returns the Hessian matrix of the log partial likelihood
        function evaluated at `params`.
        """
        pass

    def breslow_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times.
        """
        pass

    def efron_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
        pass

    def breslow_gradient(self, params):
        """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times.
        """
        pass

    def efron_gradient(self, params):
        """
        Returns the gradient of the log partial likelihood evaluated
        at `params`, using the Efron method to handle tied times.
        """
        pass

    def breslow_hessian(self, params):
        """
        Returns the Hessian of the log partial likelihood evaluated at
        `params`, using the Breslow method to handle tied times.
        """
        pass

    def efron_hessian(self, params):
        """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
        pass

    def robust_covariance(self, params):
        """
        Returns a covariance matrix for the proportional hazards model
        regresion coefficient estimates that is robust to certain
        forms of model misspecification.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the covariance matrix is
            calculated.

        Returns
        -------
        The robust covariance matrix as a square ndarray.

        Notes
        -----
        This function uses the `groups` argument to determine groups
        within which observations may be dependent.  The covariance
        matrix is calculated using the Huber-White "sandwich" approach.
        """
        pass

    def score_residuals(self, params):
        """
        Returns the score residuals calculated at a given vector of
        parameters.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the score residuals are
            calculated.

        Returns
        -------
        The score residuals, returned as a ndarray having the same
        shape as `exog`.

        Notes
        -----
        Observations in a stratum with no observed events have undefined
        score residuals, and contain NaN in the returned matrix.
        """
        pass

    def weighted_covariate_averages(self, params):
        """
        Returns the hazard-weighted average of covariate values for
        subjects who are at-risk at a particular time.

        Parameters
        ----------
        params : ndarray
            Parameter vector

        Returns
        -------
        averages : list of ndarrays
            averages[stx][i,:] is a row vector containing the weighted
            average values (for all the covariates) of at-risk
            subjects a the i^th largest observed failure time in
            stratum `stx`, using the hazard multipliers as weights.

        Notes
        -----
        Used to calculate leverages and score residuals.
        """
        pass

    def baseline_cumulative_hazard(self, params):
        """
        Estimate the baseline cumulative hazard and survival
        functions.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A list of triples (time, hazard, survival) containing the time
        values and corresponding cumulative hazard and survival
        function values for each stratum.

        Notes
        -----
        Uses the Nelson-Aalen estimator.
        """
        pass

    def baseline_cumulative_hazard_function(self, params):
        """
        Returns a function that calculates the baseline cumulative
        hazard function for each stratum.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A dict mapping stratum names to the estimated baseline
        cumulative hazard function.
        """
        pass

    def get_distribution(self, params, scale=1.0, exog=None):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Parameters
        ----------
        params : array_like
            The proportional hazards model parameters.
        scale : float
            Present for compatibility, not used.
        exog : array_like
            A design matrix, defaults to model.exog.

        Returns
        -------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes
        -----
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times within a stratum.
        """
        pass

class PHRegResults(base.LikelihoodModelResults):
    """
    Class to contain results of fitting a Cox proportional hazards
    survival model.

    PHregResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        PHreg model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        The coefficients of the fitted model.  Each coefficient is the
        log hazard ratio corresponding to a 1 unit difference in a
        single covariate while holding the other covariates fixed.
    bse : ndarray
        The standard errors of the fitted parameters.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params, scale=1.0, covariance_type='naive'):
        self.covariance_type = covariance_type
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        super(PHRegResults, self).__init__(model, params, scale=1.0, normalized_cov_params=cov_params)

    @cache_readonly
    def standard_errors(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        pass

    @cache_readonly
    def bse(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        pass

    def get_distribution(self):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Returns
        -------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes
        -----
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times within a stratum.
        """
        pass

    def _group_stats(self, groups):
        """
        Descriptive statistics of the groups.
        """
        pass

    @cache_readonly
    def weighted_covariate_averages(self):
        """
        The average covariate values within the at-risk set at each
        event time point, weighted by hazard.
        """
        pass

    @cache_readonly
    def score_residuals(self):
        """
        A matrix containing the score residuals.
        """
        pass

    @cache_readonly
    def baseline_cumulative_hazard(self):
        """
        A list (corresponding to the strata) containing the baseline
        cumulative hazard function evaluated at the event points.
        """
        pass

    @cache_readonly
    def baseline_cumulative_hazard_function(self):
        """
        A list (corresponding to the strata) containing function
        objects that calculate the cumulative hazard function.
        """
        pass

    @cache_readonly
    def schoenfeld_residuals(self):
        """
        A matrix containing the Schoenfeld residuals.

        Notes
        -----
        Schoenfeld residuals for censored observations are set to zero.
        """
        pass

    @cache_readonly
    def martingale_residuals(self):
        """
        The martingale residuals.
        """
        pass

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the proportional hazards regression results.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `x#` for ## in p the
            number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        pass

class rv_discrete_float:
    """
    A class representing a collection of discrete distributions.

    Parameters
    ----------
    xk : 2d array_like
        The support points, should be non-decreasing within each
        row.
    pk : 2d array_like
        The probabilities, should sum to one within each row.

    Notes
    -----
    Each row of `xk`, and the corresponding row of `pk` describe a
    discrete distribution.

    `xk` and `pk` should both be two-dimensional ndarrays.  Each row
    of `pk` should sum to 1.

    This class is used as a substitute for scipy.distributions.
    rv_discrete, since that class does not allow non-integer support
    points, or vectorized operations.

    Only a limited number of methods are implemented here compared to
    the other scipy distribution classes.
    """

    def __init__(self, xk, pk):
        self.xk = xk
        self.pk = pk
        self.cpk = np.cumsum(self.pk, axis=1)

    def rvs(self, n=None):
        """
        Returns a random sample from the discrete distribution.

        A vector is returned containing a single draw from each row of
        `xk`, using the probabilities of the corresponding row of `pk`

        Parameters
        ----------
        n : not used
            Present for signature compatibility
        """
        pass

    def mean(self):
        """
        Returns a vector containing the mean values of the discrete
        distributions.

        A vector is returned containing the mean value of each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """
        pass

    def var(self):
        """
        Returns a vector containing the variances of the discrete
        distributions.

        A vector is returned containing the variance for each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """
        pass

    def std(self):
        """
        Returns a vector containing the standard deviations of the
        discrete distributions.

        A vector is returned containing the standard deviation for
        each row of `xk`, using the probabilities in the corresponding
        row of `pk`.
        """
        pass