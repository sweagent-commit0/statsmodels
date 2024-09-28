"""
Dynamic factor model

Author: Chad Fulton
License: Simplified-BSD
"""
import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import is_invertible, prepare_exog, constrain_stationary_univariate, unconstrain_stationary_univariate, constrain_stationary_multivariate, unconstrain_stationary_multivariate
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender

class DynamicFactor(MLEModel):
    """
    Dynamic factor model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the observation error term,
        where "unstructured" puts no restrictions on the matrix, "diagonal"
        requires it to be any diagonal matrix (uncorrelated errors), and
        "scalar" requires it to be a scalar times the identity matrix. Default
        is "diagonal".
    error_order : int, optional
        The order of the vector autoregression followed by the observation
        error component. Default is None, corresponding to white noise errors.
    error_var : bool, optional
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'diagonal', 'unstructured'}
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors).
    error_order : int
        The order of the vector autoregression followed by the observation
        error component.
    error_var : bool
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.

    Notes
    -----
    The dynamic factor model considered here is in the so-called static form,
    and is specified:

    .. math::

        y_t & = \\Lambda f_t + B x_t + u_t \\\\
        f_t & = A_1 f_{t-1} + \\dots + A_p f_{t-p} + \\eta_t \\\\
        u_t & = C_1 u_{t-1} + \\dots + C_q u_{t-q} + \\varepsilon_t

    where there are `k_endog` observed series and `k_factors` unobserved
    factors. Thus :math:`y_t` is a `k_endog` x 1 vector and :math:`f_t` is a
    `k_factors` x 1 vector.

    :math:`x_t` are optional exogenous vectors, shaped `k_exog` x 1.

    :math:`\\eta_t` and :math:`\\varepsilon_t` are white noise error terms. In
    order to identify the factors, :math:`Var(\\eta_t) = I`. Denote
    :math:`Var(\\varepsilon_t) \\equiv \\Sigma`.

    Options related to the unobserved factors:

    - `k_factors`: this is the dimension of the vector :math:`f_t`, above.
      To exclude factors completely, set `k_factors = 0`.
    - `factor_order`: this is the number of lags to include in the factor
      evolution equation, and corresponds to :math:`p`, above. To have static
      factors, set `factor_order = 0`.

    Options related to the observation error term :math:`u_t`:

    - `error_order`: the number of lags to include in the error evolution
      equation; corresponds to :math:`q`, above. To have white noise errors,
      set `error_order = 0` (this is the default).
    - `error_cov_type`: this controls the form of the covariance matrix
      :math:`\\Sigma`. If it is "dscalar", then :math:`\\Sigma = \\sigma^2 I`. If
      it is "diagonal", then
      :math:`\\Sigma = \\text{diag}(\\sigma_1^2, \\dots, \\sigma_n^2)`. If it is
      "unstructured", then :math:`\\Sigma` is any valid variance / covariance
      matrix (i.e. symmetric and positive definite).
    - `error_var`: this controls whether or not the errors evolve jointly
      according to a VAR(q), or individually according to separate AR(q)
      processes. In terms of the formulation above, if `error_var = False`,
      then the matrices :math:C_i` are diagonal, otherwise they are general
      VAR matrices.

    References
    ----------
    .. [*] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.
    """

    def __init__(self, endog, k_factors, factor_order, exog=None, error_order=0, error_var=False, error_cov_type='diagonal', enforce_stationarity=True, **kwargs):
        self.enforce_stationarity = enforce_stationarity
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var and error_order > 0
        self.error_cov_type = error_cov_type
        self.k_exog, exog = prepare_exog(exog)
        self.mle_regression = self.k_exog > 0
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog, order='C')
        k_endog = endog.shape[1] if endog.ndim > 1 else 1
        self._factor_order = max(1, self.factor_order) * self.k_factors
        self._error_order = self.error_order * k_endog
        k_states = self._factor_order
        k_posdef = self.k_factors
        if self.error_order > 0:
            k_states += self._error_order
            k_posdef += k_endog
        self._unused_state = False
        if k_states == 0:
            k_states = 1
            k_posdef = 1
            self._unused_state = True
        if k_endog < 2:
            raise ValueError('The dynamic factors model is only valid for multivariate time series.')
        if self.k_factors >= k_endog:
            raise ValueError('Number of factors must be less than the number of endogenous variables.')
        if self.error_cov_type not in ['scalar', 'diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type specification.')
        kwargs.setdefault('initialization', 'stationary')
        super(DynamicFactor, self).__init__(endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        if self.k_exog > 0:
            self.ssm._time_invariant = False
        self.parameters = {}
        self._initialize_loadings()
        self._initialize_exog()
        self._initialize_error_cov()
        self._initialize_factor_transition()
        self._initialize_error_transition()
        self.k_params = sum(self.parameters.values())

        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return (param_slice, offset)
        offset = 0
        self._params_loadings, offset = _slice('factor_loadings', offset)
        self._params_exog, offset = _slice('exog', offset)
        self._params_error_cov, offset = _slice('error_cov', offset)
        self._params_factor_transition, offset = _slice('factor_transition', offset)
        self._params_error_transition, offset = _slice('error_transition', offset)
        self._init_keys += ['k_factors', 'factor_order', 'error_order', 'error_var', 'error_cov_type', 'enforce_stationarity'] + list(kwargs.keys())

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        Constrains the factor transition to be stationary and variances to be
        positive.
        """
        pass

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        pass

    def update(self, params, transformed=True, includes_fixed=False, complex_step=False):
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

        Notes
        -----
        Let `n = k_endog`, `m = k_factors`, and `p = factor_order`. Then the
        `params` vector has length
        :math:`[n 	imes m] + [n] + [m^2 	imes p]`.
        It is expanded in the following way:

        - The first :math:`n 	imes m` parameters fill out the factor loading
          matrix, starting from the [0,0] entry and then proceeding along rows.
          These parameters are not modified in `transform_params`.
        - The next :math:`n` parameters provide variances for the error_cov
          errors in the observation equation. They fill in the diagonal of the
          observation covariance matrix, and are constrained to be positive by
          `transofrm_params`.
        - The next :math:`m^2 	imes p` parameters are used to create the `p`
          coefficient matrices for the vector autoregression describing the
          factor transition. They are transformed in `transform_params` to
          enforce stationarity of the VAR(p). They are placed so as to make
          the transition matrix a companion matrix for the VAR. In particular,
          we assume that the first :math:`m^2` parameters fill the first
          coefficient matrix (starting at [0,0] and filling along rows), the
          second :math:`m^2` parameters fill the second matrix, etc.
        """
        pass

class DynamicFactorResults(MLEResults):
    """
    Class to hold results from fitting an DynamicFactor model.

    Parameters
    ----------
    model : DynamicFactor instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the DynamicFactor model
        instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super(DynamicFactorResults, self).__init__(model, params, filter_results, cov_type, **kwargs)
        self.df_resid = np.inf
        self.specification = Bunch(**{'k_endog': self.model.k_endog, 'enforce_stationarity': self.model.enforce_stationarity, 'k_factors': self.model.k_factors, 'factor_order': self.model.factor_order, 'error_order': self.model.error_order, 'error_var': self.model.error_var, 'error_cov_type': self.model.error_cov_type, 'k_exog': self.model.k_exog})
        self.coefficient_matrices_var = None
        if self.model.factor_order > 0:
            ar_params = np.array(self.params[self.model._params_factor_transition])
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            self.coefficient_matrices_var = ar_params.reshape(k_factors * factor_order, k_factors).T.reshape(k_factors, k_factors, factor_order).T
        self.coefficient_matrices_error = None
        if self.model.error_order > 0:
            ar_params = np.array(self.params[self.model._params_error_transition])
            k_endog = self.model.k_endog
            error_order = self.model.error_order
            if self.model.error_var:
                self.coefficient_matrices_error = ar_params.reshape(k_endog * error_order, k_endog).T.reshape(k_endog, k_endog, error_order).T
            else:
                mat = np.zeros((k_endog, k_endog * error_order))
                mat[self.model._idx_error_diag] = ar_params
                self.coefficient_matrices_error = mat.T.reshape(error_order, k_endog, k_endog)

    @property
    def factors(self):
        """
        Estimates of unobserved factors

        Returns
        -------
        out : Bunch
            Has the following attributes shown in Notes.

        Notes
        -----
        The output is a bunch of the following format:

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
    def coefficients_of_determination(self):
        """
        Coefficients of determination (:math:`R^2`) from regressions of
        individual estimated factors on endogenous variables.

        Returns
        -------
        coefficients_of_determination : ndarray
            A `k_endog` x `k_factors` array, where
            `coefficients_of_determination[i, j]` represents the :math:`R^2`
            value from a regression of factor `j` and a constant on endogenous
            variable `i`.

        Notes
        -----
        Although it can be difficult to interpret the estimated factor loadings
        and factors, it is often helpful to use the coefficients of
        determination from univariate regressions to assess the importance of
        each factor in explaining the variation in each endogenous variable.

        In models with many variables and factors, this can sometimes lend
        interpretation to the factors (for example sometimes one factor will
        load primarily on real variables and another on nominal variables).

        See Also
        --------
        plot_coefficients_of_determination
        """
        pass

    def plot_coefficients_of_determination(self, endog_labels=None, fig=None, figsize=None):
        """
        Plot the coefficients of determination

        Parameters
        ----------
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----

        Produces a `k_factors` x 1 plot grid. The `i`th plot shows a bar plot
        of the coefficients of determination associated with factor `i`. The
        endogenous variables are arranged along the x-axis according to their
        position in the `endog` array.

        See Also
        --------
        coefficients_of_determination
        """
        pass

class DynamicFactorResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(DynamicFactorResultsWrapper, DynamicFactorResults)