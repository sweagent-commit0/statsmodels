"""
Recursive least squares model

Author: Chad Fulton
License: Simplified-BSD
"""
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults, MLEResultsWrapper, PredictionResults, PredictionResultsWrapper
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
_cusum_squares_scalars = np.array([[1.072983, 1.2238734, 1.3581015, 1.5174271, 1.6276236], [-0.6698868, -0.6700069, -0.6701218, -0.6702672, -0.6703724], [-0.5816458, -0.7351697, -0.8858694, -1.0847745, -1.2365861]])

class RecursiveLS(MLEModel):
    """
    Recursive least squares

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like
        Array of exogenous regressors, shaped nobs x k.
    constraints : array_like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

    Notes
    -----
    Recursive least squares (RLS) corresponds to expanding window ordinary
    least squares (OLS).

    This model applies the Kalman filter to compute recursive estimates of the
    coefficients and recursive residuals.

    References
    ----------
    .. [*] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, endog, exog, constraints=None, **kwargs):
        endog_using_pandas = _is_using_pandas(endog, None)
        if not endog_using_pandas:
            endog = np.asanyarray(endog)
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)
        self.k_exog = exog.shape[1]
        self.k_constraints = 0
        self._r_matrix = self._q_matrix = None
        if constraints is not None:
            from patsy import DesignInfo
            from statsmodels.base.data import handle_data
            data = handle_data(endog, exog, **kwargs)
            names = data.param_names
            LC = DesignInfo(names).linear_constraint(constraints)
            self._r_matrix, self._q_matrix = (LC.coefs, LC.constants)
            self.k_constraints = self._r_matrix.shape[0]
            nobs = len(endog)
            constraint_endog = np.zeros((nobs, len(self._r_matrix)))
            if endog_using_pandas:
                constraint_endog = pd.DataFrame(constraint_endog, index=endog.index)
                endog = concat([endog, constraint_endog], axis=1)
                endog.iloc[:, 1:] = np.tile(self._q_matrix.T, (nobs, 1))
            else:
                endog[:, 1:] = self._q_matrix[:, 0]
        kwargs.setdefault('initialization', 'diffuse')
        formula_kwargs = ['missing', 'missing_idx', 'formula', 'design_info']
        for name in formula_kwargs:
            if name in kwargs:
                del kwargs[name]
        super(RecursiveLS, self).__init__(endog, k_states=self.k_exog, exog=exog, **kwargs)
        self.ssm.filter_univariate = True
        self.ssm.filter_concentrated = True
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['design', 0] = self.exog[:, :, None].T
        if self._r_matrix is not None:
            self['design', 1:, :] = self._r_matrix[:, :, None]
        self['transition'] = np.eye(self.k_states)
        self['obs_cov', 0, 0] = 1.0
        self['transition'] = np.eye(self.k_states)
        if self._r_matrix is not None:
            self.k_endog = 1

    def fit(self):
        """
        Fits the model by application of the Kalman filter

        Returns
        -------
        RecursiveLSResults
        """
        pass

    def update(self, params, **kwargs):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.
        """
        pass

class RecursiveLSResults(MLEResults):
    """
    Class to hold results from fitting a recursive least squares model.

    Parameters
    ----------
    model : RecursiveLS instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the recursive least squares
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg', **kwargs):
        super(RecursiveLSResults, self).__init__(model, params, filter_results, cov_type, **kwargs)
        q = max(self.loglikelihood_burn, self.k_diffuse_states)
        self.df_model = q - self.model.k_constraints
        self.df_resid = self.nobs_effective - self.df_model
        self._init_kwds = self.model._get_init_kwds()
        self.specification = Bunch(**{'k_exog': self.model.k_exog, 'k_constraints': self.model.k_constraints})
        if self.model._r_matrix is not None:
            for name in ['forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov']:
                setattr(self, name, getattr(self, name)[0:1])

    @property
    def recursive_coefficients(self):
        """
        Estimates of regression coefficients, recursively estimated

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        pass

    @cache_readonly
    def resid_recursive(self):
        """
        Recursive residuals

        Returns
        -------
        resid_recursive : array_like
            An array of length `nobs` holding the recursive
            residuals.

        Notes
        -----
        These quantities are defined in, for example, Harvey (1989)
        section 5.4. In fact, there he defines the standardized innovations in
        equation 5.4.1, but in his version they have non-unit variance, whereas
        the standardized forecast errors computed by the Kalman filter here
        assume unit variance. To convert to Harvey's definition, we need to
        multiply by the standard deviation.

        Harvey notes that in smaller samples, "although the second moment
        of the :math:`\\tilde \\sigma_*^{-1} \\tilde v_t`'s is unity, the
        variance is not necessarily equal to unity as the mean need not be
        equal to zero", and he defines an alternative version (which are
        not provided here).
        """
        pass

    @cache_readonly
    def cusum(self):
        """
        Cumulative sum of standardized recursive residuals statistics

        Returns
        -------
        cusum : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM statistics.

        Notes
        -----
        The CUSUM statistic takes the form:

        .. math::

            W_t = \\frac{1}{\\hat \\sigma} \\sum_{j=k+1}^t w_j

        where :math:`w_j` is the recursive residual at time :math:`j` and
        :math:`\\hat \\sigma` is the estimate of the standard deviation
        from the full sample.

        Excludes the first `k_exog` datapoints.

        Due to differences in the way :math:`\\hat \\sigma` is calculated, the
        output of this function differs slightly from the output in the
        R package strucchange and the Stata contributed .ado file cusum6. The
        calculation in this package is consistent with the description of
        Brown et al. (1975)

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        pass

    @cache_readonly
    def cusum_squares(self):
        """
        Cumulative sum of squares of standardized recursive residuals
        statistics

        Returns
        -------
        cusum_squares : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM of squares statistics.

        Notes
        -----
        The CUSUM of squares statistic takes the form:

        .. math::

            s_t = \\left ( \\sum_{j=k+1}^t w_j^2 \\right ) \\Bigg /
                  \\left ( \\sum_{j=k+1}^T w_j^2 \\right )

        where :math:`w_j` is the recursive residual at time :math:`j`.

        Excludes the first `k_exog` datapoints.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        pass

    @cache_readonly
    def llf_recursive_obs(self):
        """
        (float) Loglikelihood at observation, computed from recursive residuals
        """
        pass

    @cache_readonly
    def llf_recursive(self):
        """
        (float) Loglikelihood defined by recursive residuals, equivalent to OLS
        """
        pass

    @cache_readonly
    def ssr(self):
        """ssr"""
        pass

    @cache_readonly
    def centered_tss(self):
        """Centered tss"""
        pass

    @cache_readonly
    def uncentered_tss(self):
        """uncentered tss"""
        pass

    @cache_readonly
    def ess(self):
        """ess"""
        pass

    @cache_readonly
    def rsquared(self):
        """rsquared"""
        pass

    @cache_readonly
    def mse_model(self):
        """mse_model"""
        pass

    @cache_readonly
    def mse_resid(self):
        """mse_resid"""
        pass

    @cache_readonly
    def mse_total(self):
        """mse_total"""
        pass

    def plot_recursive_coefficient(self, variables=0, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the recursively estimated coefficients on a given variable

        Parameters
        ----------
        variables : {int, str, list[int], list[str]}, optional
            Integer index or string name of the variable whose coefficient will
            be plotted. Can also be an iterable of integers or strings. Default
            is the first variable.
        alpha : float, optional
            The confidence intervals for the coefficient are (1 - alpha) %
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        pass

    def _cusum_significance_bounds(self, alpha, ddof=0, points=None):
        """
        Parameters
        ----------
        alpha : float, optional
            The significance bound is alpha %.
        ddof : int, optional
            The number of periods additional to `k_exog` to exclude in
            constructing the bounds. Default is zero. This is usually used
            only for testing purposes.
        points : iterable, optional
            The points at which to evaluate the significance bounds. Default is
            two points, beginning and end of the sample.

        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lw, uw) because they burn the first k_exog + 1 periods instead of the
        first k_exog. If this change is performed
        (so that `tmp = (self.nobs - d - 1)**0.5`), then the output here
        matches cusum6.

        The cusum6 behavior does not seem to be consistent with
        Brown et al. (1975); it is likely they did that because they needed
        three initial observations to get the initial OLS estimates, whereas
        we do not need to do that.
        """
        pass

    def plot_cusum(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the CUSUM statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM statistic
        moves out of the significance bounds.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
        pass

    def _cusum_squares_significance_bounds(self, alpha, points=None):
        """
        Notes
        -----
        Comparing against the cusum6 package for Stata, this does not produce
        exactly the same confidence bands (which are produced in cusum6 by
        lww, uww) because they use a different method for computing the
        critical value; in particular, they use tabled values from
        Table C, pp. 364-365 of "The Econometric Analysis of Time Series"
        Harvey, (1990), and use the value given to 99 observations for any
        larger number of observations. In contrast, we use the approximating
        critical values suggested in Edgerton and Wells (1994) which allows
        computing relatively good approximations for any number of
        observations.
        """
        pass

    def plot_cusum_squares(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
        """
        Plot the CUSUM of squares statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM of squares
        statistic moves out of the significance bounds.

        Critical values used in creating the significance bounds are computed
        using the approximate formula of [1]_.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        .. [1] Edgerton, David, and Curt Wells. 1994.
           "Critical Values for the Cusumsq Statistic
           in Medium and Large Sized Samples."
           Oxford Bulletin of Economics and Statistics 56 (3): 355-65.
        """
        pass

class RecursiveLSResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(RecursiveLSResultsWrapper, RecursiveLSResults)