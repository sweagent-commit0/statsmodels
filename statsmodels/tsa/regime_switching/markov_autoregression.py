"""
Markov switching autoregression models

Author: Chad Fulton
License: BSD-3
"""
import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import markov_switching, markov_regression
from statsmodels.tsa.statespace.tools import constrain_stationary_univariate, unconstrain_stationary_univariate

class MarkovAutoregression(markov_regression.MarkovRegression):
    """
    Markov switching regression model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    order : int
        The order of the autoregressive lag polynomial.
    trend : {'n', 'c', 't', 'ct'}
        Whether or not to include a trend. To include an constant, time trend,
        or both, set `trend='c'`, `trend='t'`, or `trend='ct'`. For no trend,
        set `trend='n'`. Default is a constant.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.
    switching_ar : bool or iterable, optional
        If a boolean, sets whether or not all autoregressive coefficients are
        switching across regimes. If an iterable, should be of length equal
        to `order`, where each element is a boolean describing whether the
        corresponding coefficient is switching. Default is True.
    switching_trend : bool or iterable, optional
        If a boolean, sets whether or not all trend coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of trend variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_exog : bool or iterable, optional
        If a boolean, sets whether or not all regression coefficients are
        switching across regimes. If an iterable, should be of length equal
        to the number of exogenous variables, where each element is
        a boolean describing whether the corresponding coefficient is
        switching. Default is True.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        False.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    The model can be written as:

    .. math::

        y_t = a_{S_t} + x_t' \\beta_{S_t} + \\phi_{1, S_t}
        (y_{t-1} - a_{S_{t-1}} - x_{t-1}' \\beta_{S_{t-1}}) + \\dots +
        \\phi_{p, S_t} (y_{t-p} - a_{S_{t-p}} - x_{t-p}' \\beta_{S_{t-p}}) +
        \\varepsilon_t \\\\
        \\varepsilon_t \\sim N(0, \\sigma_{S_t}^2)

    i.e. the model is an autoregression with where the autoregressive
    coefficients, the mean of the process (possibly including trend or
    regression effects) and the variance of the error term may be switching
    across regimes.

    The `trend` is accommodated by prepending columns to the `exog` array. Thus
    if `trend='c'`, the passed `exog` array should not already have a column of
    ones.

    See the notebook `Markov switching autoregression
    <../examples/notebooks/generated/markov_autoregression.html>`__
    for an overview.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, endog, k_regimes, order, trend='c', exog=None, exog_tvtp=None, switching_ar=True, switching_trend=True, switching_exog=False, switching_variance=False, dates=None, freq=None, missing='none'):
        self.switching_ar = switching_ar
        if self.switching_ar is True or self.switching_ar is False:
            self.switching_ar = [self.switching_ar] * order
        elif not len(self.switching_ar) == order:
            raise ValueError('Invalid iterable passed to `switching_ar`.')
        super().__init__(endog, k_regimes, trend=trend, exog=exog, order=order, exog_tvtp=exog_tvtp, switching_trend=switching_trend, switching_exog=switching_exog, switching_variance=switching_variance, dates=dates, freq=freq, missing=missing)
        if self.nobs <= self.order:
            raise ValueError('Must have more observations than the order of the autoregression.')
        self.exog_ar = lagmat(endog, self.order)[self.order:]
        self.nobs -= self.order
        self.orig_endog = self.endog
        self.endog = self.endog[self.order:]
        if self._k_exog > 0:
            self.orig_exog = self.exog
            self.exog = self.exog[self.order:]
        self.data.endog, self.data.exog = self.data._convert_endog_exog(self.endog, self.exog)
        if self.data.row_labels is not None:
            self.data._cache['row_labels'] = self.data.row_labels[self.order:]
        if self._index is not None:
            if self._index_generated:
                self._index = self._index[:-self.order]
            else:
                self._index = self._index[self.order:]
        self.parameters['autoregressive'] = self.switching_ar
        self._predict_slices = [slice(None, None, None)] * (self.order + 1)

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current and previous regime

        Parameters
        ----------
        params : array_like
            Array of parameters at which to create predictions.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        pass

    def _conditional_loglikelihoods(self, params):
        """
        Compute loglikelihoods conditional on the current period's regime and
        the last `self.order` regimes.
        """
        pass

    def _em_iteration(self, params0):
        """
        EM iteration
        """
        pass

    def _em_autoregressive(self, result, betas, tmp=None):
        """
        EM step for autoregressive coefficients and variances
        """
        pass

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        pass

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        pass

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
        """
        pass

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

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

class MarkovAutoregressionResults(markov_regression.MarkovRegressionResults):
    """
    Class to hold results from fitting a Markov switching autoregression model

    Parameters
    ----------
    model : MarkovAutoregression instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """
    pass

class MarkovAutoregressionResultsWrapper(markov_regression.MarkovRegressionResultsWrapper):
    pass
wrap.populate_wrapper(MarkovAutoregressionResultsWrapper, MarkovAutoregressionResults)