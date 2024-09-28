"""
Markov switching models

Author: Chad Fulton
License: BSD-3
"""
import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import chamilton_filter_log, dhamilton_filter_log, shamilton_filter_log, zhamilton_filter_log
from statsmodels.tsa.regime_switching._kim_smoother import ckim_smoother_log, dkim_smoother_log, skim_smoother_log, zkim_smoother_log
from statsmodels.tsa.statespace.tools import find_best_blas_type, prepare_exog, _safe_cond
prefix_hamilton_filter_log_map = {'s': shamilton_filter_log, 'd': dhamilton_filter_log, 'c': chamilton_filter_log, 'z': zhamilton_filter_log}
prefix_kim_smoother_log_map = {'s': skim_smoother_log, 'd': dkim_smoother_log, 'c': ckim_smoother_log, 'z': zkim_smoother_log}

def _logistic(x):
    """
    Note that this is not a vectorized function
    """
    pass

def _partials_logistic(x):
    """
    Note that this is not a vectorized function
    """
    pass

def cy_hamilton_filter_log(initial_probabilities, regime_transition, conditional_loglikelihoods, model_order):
    """
    Hamilton filter in log space using Cython inner loop.

    Parameters
    ----------
    initial_probabilities : ndarray
        Array of initial probabilities, shaped (k_regimes,) giving the
        distribution of the regime process at time t = -order where order
        is a nonnegative integer.
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs + order).  Entry [i, j,
        t] contains the probability of moving from j at time t-1 to i at
        time t, so each matrix regime_transition[:, :, t] should be left
        stochastic.  The first order entries and initial_probabilities are
        used to produce the initial joint distribution of dimension order +
        1 at time t=0.
    conditional_loglikelihoods : ndarray
        Array of loglikelihoods conditional on the last `order+1` regimes,
        shaped (k_regimes,)*(order + 1) + (nobs,).

    Returns
    -------
    filtered_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_t] - the probability of being in each
        regime conditional on time t information. Shaped (k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    joint_loglikelihoods : ndarray
        Array of loglikelihoods condition on time t information,
        shaped (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    """
    pass

def cy_kim_smoother_log(regime_transition, predicted_joint_probabilities, filtered_joint_probabilities):
    """
    Kim smoother in log space using Cython inner loop.

    Parameters
    ----------
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    Returns
    -------
    smoothed_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """
    pass

class MarkovSwitchingParams:
    """
    Class to hold parameters in Markov switching models

    Parameters
    ----------
    k_regimes : int
        The number of regimes between which parameters may switch.

    Notes
    -----

    The purpose is to allow selecting parameter indexes / slices based on
    parameter type, regime number, or both.

    Parameters are lexicographically ordered in the following way:

    1. Named type string (e.g. "autoregressive")
    2. Number (e.g. the first autoregressive parameter, then the second)
    3. Regime (if applicable)

    Parameter blocks are set using dictionary setter notation where the key
    is the named type string and the value is a list of boolean values
    indicating whether a given parameter is switching or not.

    For example, consider the following code:

        parameters = MarkovSwitchingParams(k_regimes=2)
        parameters['regime_transition'] = [1,1]
        parameters['exog'] = [0, 1]

    This implies the model has 7 parameters: 4 "regime_transition"-related
    parameters (2 parameters that each switch according to regimes) and 3
    "exog"-related parameters (1 parameter that does not switch, and one 1 that
    does).

    The order of parameters is then:

    1. The first "regime_transition" parameter, regime 0
    2. The first "regime_transition" parameter, regime 1
    3. The second "regime_transition" parameter, regime 1
    4. The second "regime_transition" parameter, regime 1
    5. The first "exog" parameter
    6. The second "exog" parameter, regime 0
    7. The second "exog" parameter, regime 1

    Retrieving indexes / slices is done through dictionary getter notation.
    There are three options for the dictionary key:

    - Regime number (zero-indexed)
    - Named type string (e.g. "autoregressive")
    - Regime number and named type string

    In the above example, consider the following getters:

    >>> parameters[0]
    array([0, 2, 4, 6])
    >>> parameters[1]
    array([1, 3, 5, 6])
    >>> parameters['exog']
    slice(4, 7, None)
    >>> parameters[0, 'exog']
    [4, 6]
    >>> parameters[1, 'exog']
    [4, 7]

    Notice that in the last two examples, both lists of indexes include 4.
    That's because that is the index of the the non-switching first "exog"
    parameter, which should be selected regardless of the regime.

    In addition to the getter, the `k_parameters` attribute is an dict
    with the named type strings as the keys. It can be used to get the total
    number of parameters of each type:

    >>> parameters.k_parameters['regime_transition']
    4
    >>> parameters.k_parameters['exog']
    3
    """

    def __init__(self, k_regimes):
        self.k_regimes = k_regimes
        self.k_params = 0
        self.k_parameters = {}
        self.switching = {}
        self.slices_purpose = {}
        self.relative_index_regime_purpose = [{} for i in range(self.k_regimes)]
        self.index_regime_purpose = [{} for i in range(self.k_regimes)]
        self.index_regime = [[] for i in range(self.k_regimes)]

    def __getitem__(self, key):
        _type = type(key)
        if _type is str:
            return self.slices_purpose[key]
        elif _type is int:
            return self.index_regime[key]
        elif _type is tuple:
            if not len(key) == 2:
                raise IndexError('Invalid index')
            if type(key[1]) is str and type(key[0]) is int:
                return self.index_regime_purpose[key[0]][key[1]]
            elif type(key[0]) is str and type(key[1]) is int:
                return self.index_regime_purpose[key[1]][key[0]]
            else:
                raise IndexError('Invalid index')
        else:
            raise IndexError('Invalid index')

    def __setitem__(self, key, value):
        _type = type(key)
        if _type is str:
            value = np.array(value, dtype=bool, ndmin=1)
            k_params = self.k_params
            self.k_parameters[key] = value.size + np.sum(value) * (self.k_regimes - 1)
            self.k_params += self.k_parameters[key]
            self.switching[key] = value
            self.slices_purpose[key] = np.s_[k_params:self.k_params]
            for j in range(self.k_regimes):
                self.relative_index_regime_purpose[j][key] = []
                self.index_regime_purpose[j][key] = []
            offset = 0
            for i in range(value.size):
                switching = value[i]
                for j in range(self.k_regimes):
                    if not switching:
                        self.relative_index_regime_purpose[j][key].append(offset)
                    else:
                        self.relative_index_regime_purpose[j][key].append(offset + j)
                offset += 1 if not switching else self.k_regimes
            for j in range(self.k_regimes):
                offset = 0
                indices = []
                for k, v in self.relative_index_regime_purpose[j].items():
                    v = (np.r_[v] + offset).tolist()
                    self.index_regime_purpose[j][k] = v
                    indices.append(v)
                    offset += self.k_parameters[k]
                self.index_regime[j] = np.concatenate(indices).astype(int)
        else:
            raise IndexError('Invalid index')

class MarkovSwitching(tsbase.TimeSeriesModel):
    """
    First-order k-regime Markov switching model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    order : int, optional
        The order of the model describes the dependence of the likelihood on
        previous regimes. This depends on the model in question and should be
        set appropriately by subclasses.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, endog, k_regimes, order=0, exog_tvtp=None, exog=None, dates=None, freq=None, missing='none'):
        self.k_regimes = k_regimes
        self.tvtp = exog_tvtp is not None
        self.order = order
        self.k_tvtp, self.exog_tvtp = prepare_exog(exog_tvtp)
        super(MarkovSwitching, self).__init__(endog, exog, dates=dates, freq=freq, missing=missing)
        self.nobs = self.endog.shape[0]
        if self.endog.ndim > 1 and self.endog.shape[1] > 1:
            raise ValueError('Must have univariate endogenous data.')
        if self.k_regimes < 2:
            raise ValueError('Markov switching models must have at least two regimes.')
        if not (self.exog_tvtp is None or self.exog_tvtp.shape[0] == self.nobs):
            raise ValueError('Time-varying transition probabilities exogenous array must have the same number of observations as the endogenous array.')
        self.parameters = MarkovSwitchingParams(self.k_regimes)
        k_transition = self.k_regimes - 1
        if self.tvtp:
            k_transition *= self.k_tvtp
        self.parameters['regime_transition'] = [1] * k_transition
        self._initialization = 'steady-state'
        self._initial_probabilities = None

    @property
    def k_params(self):
        """
        (int) Number of parameters in the model
        """
        pass

    def initialize_steady_state(self):
        """
        Set initialization of regime probabilities to be steady-state values

        Notes
        -----
        Only valid if there are not time-varying transition probabilities.
        """
        pass

    def initialize_known(self, probabilities, tol=1e-08):
        """
        Set initialization of regime probabilities to use known values
        """
        pass

    def initial_probabilities(self, params, regime_transition=None):
        """
        Retrieve initial probabilities
        """
        pass

    def regime_transition_matrix(self, params, exog_tvtp=None):
        """
        Construct the left-stochastic transition matrix

        Notes
        -----
        This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
        are time-varying transition probabilities, it will be shaped
        (k_regimes, k_regimes, nobs).

        The (i,j)th element of this matrix is the probability of transitioning
        from regime j to regime i; thus the previous regime is represented in a
        column and the next regime is represented by a row.

        It is left-stochastic, meaning that each column sums to one (because
        it is certain that from one regime (j) you will transition to *some
        other regime*).
        """
        pass

    def predict(self, params, start=None, end=None, probabilities=None, conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        params : ndarray
            Parameters at which to form predictions
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        probabilities : str or array_like, optional
            Specifies the weighting probabilities used in constructing the
            prediction as a weighted average. If a string, can be 'predicted',
            'filtered', or 'smoothed'. Otherwise can be an array of
            probabilities to use. Default is smoothed.
        conditional : bool or int, optional
            Whether or not to return predictions conditional on current or
            past regimes. If False, returns a single vector of weighted
            predictions. If True or 1, returns predictions conditional on the
            current regime. For larger integers, returns predictions
            conditional on the current regime and some number of past regimes.

        Returns
        -------
        predict : ndarray
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        pass

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current, and possibly past,
        regimes

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform prediction.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        pass

    def _conditional_loglikelihoods(self, params):
        """
        Compute likelihoods conditional on the current period's regime (and
        the last self.order periods' regimes if self.order > 0).

        Must be implemented in subclasses.
        """
        pass

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None, return_raw=False, results_class=None, results_wrapper_class=None):
        """
        Apply the Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : bool,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        pass

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None, return_raw=False, results_class=None, results_wrapper_class=None):
        """
        Apply the Kim smoother and Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : bool,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        pass

    def loglikeobs(self, params, transformed=True):
        """
        Loglikelihood evaluation for each period

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        pass

    def loglike(self, params, transformed=True):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        pass

    def score(self, params, transformed=True):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        pass

    def score_obs(self, params, transformed=True):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        pass

    def hessian(self, params, transformed=True):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the Hessian
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        pass

    def fit(self, start_params=None, transformed=True, cov_type='approx', cov_kwds=None, method='bfgs', maxiter=100, full_output=1, disp=0, callback=None, return_params=False, em_iter=5, search_reps=0, search_iter=5, search_scale=1.0, **kwargs):
        """
        Fits the model by maximum likelihood via Hamilton filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'approx'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        em_iter : int, optional
            Number of initial EM iteration steps used to improve starting
            parameters.
        search_reps : int, optional
            Number of randomly drawn search parameters that are drawn around
            `start_params` to try and improve starting parameters. Default is
            0.
        search_iter : int, optional
            Number of initial EM iteration steps used to improve each of the
            search parameter repetitions.
        search_scale : float or array, optional.
            Scale of variates for random start parameter search.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MarkovSwitchingResults
        """
        pass

    def _fit_em(self, start_params=None, transformed=True, cov_type='none', cov_kwds=None, maxiter=50, tolerance=1e-06, full_output=True, return_params=False, **kwargs):
        """
        Fits the model using the Expectation-Maximization (EM) algorithm

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by `start_params`.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'none'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        maxiter : int, optional
            The maximum number of iterations to perform.
        tolerance : float, optional
            The iteration stops when the difference between subsequent
            loglikelihood values is less than this tolerance.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. This includes all intermediate values for
            parameters and loglikelihood values
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring. It has not been tested for a thoroughly correct EM
        implementation in all cases. It does not support TVTP transition
        probabilities.

        Returns
        -------
        MarkovSwitchingResults
        """
        pass

    def _em_iteration(self, params0):
        """
        EM iteration

        Notes
        -----
        The EM iteration in this base class only performs the EM step for
        non-TVTP transition probabilities.
        """
        pass

    def _em_regime_transition(self, result):
        """
        EM step for regime transition probabilities
        """
        pass

    def _start_params_search(self, reps, start_params=None, transformed=True, em_iter=5, scale=1.0):
        """
        Search for starting parameters as random permutations of a vector

        Parameters
        ----------
        reps : int
            Number of random permutations to try.
        start_params : ndarray, optional
            Starting parameter vector. If not given, class-level start
            parameters are used.
        transformed : bool, optional
            If `start_params` was provided, whether or not those parameters
            are already transformed. Default is True.
        em_iter : int, optional
            Number of EM iterations to apply to each random permutation.
        scale : array or float, optional
            Scale of variates for random start parameter search. Can be given
            as an array of length equal to the number of parameters or as a
            single scalar.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring, where the defaults have been set heuristically.
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

        Notes
        -----
        In the base class, this only transforms the transition-probability-
        related parameters.
        """
        pass

    def _untransform_logistic(self, unconstrained, constrained):
        """
        Function to allow using a numerical root-finder to reverse the
        logistic transform.
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

        Notes
        -----
        In the base class, this only untransforms the transition-probability-
        related parameters.
        """
        pass

class HamiltonFilterResults:
    """
    Results from applying the Hamilton filter to a state space model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_regimes : int
        The number of unobserved regimes.
    regime_transition : ndarray
        The regime transition matrix.
    initialization : str
        Initialization method for regime probabilities.
    initial_probabilities : ndarray
        Initial regime probabilities
    conditional_loglikelihoods : ndarray
        The loglikelihood values at each time period, conditional on regime.
    predicted_joint_probabilities : ndarray
        Predicted joint probabilities at each time period.
    filtered_marginal_probabilities : ndarray
        Filtered marginal probabilities at each time period.
    filtered_joint_probabilities : ndarray
        Filtered joint probabilities at each time period.
    joint_loglikelihoods : ndarray
        The likelihood values at each time period.
    llf_obs : ndarray
        The loglikelihood values at each time period.
    """

    def __init__(self, model, result):
        self.model = model
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes
        attributes = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'predicted_joint_probabilities', 'filtered_marginal_probabilities', 'filtered_joint_probabilities', 'joint_loglikelihoods']
        for name in attributes:
            setattr(self, name, getattr(result, name))
        self.initialization = model._initialization
        self.llf_obs = self.joint_loglikelihoods
        self.llf = np.sum(self.llf_obs)
        if self.regime_transition.shape[-1] > 1 and self.order > 0:
            self.regime_transition = self.regime_transition[..., self.order:]
        self._predicted_marginal_probabilities = None

    @property
    def expected_durations(self):
        """
        (array) Expected duration of a regime, possibly time-varying.
        """
        pass

class KimSmootherResults(HamiltonFilterResults):
    """
    Results from applying the Kim smoother to a Markov switching model.

    Parameters
    ----------
    model : MarkovSwitchingModel
        The model object.
    result : dict
        A dictionary containing two keys: 'smoothd_joint_probabilities' and
        'smoothed_marginal_probabilities'.

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    """

    def __init__(self, model, result):
        super(KimSmootherResults, self).__init__(model, result)
        attributes = ['smoothed_joint_probabilities', 'smoothed_marginal_probabilities']
        for name in attributes:
            setattr(self, name, getattr(result, name))

class MarkovSwitchingResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting a Markov switching model

    Parameters
    ----------
    model : MarkovSwitching instance
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
    use_t = False

    def __init__(self, model, params, results, cov_type='opg', cov_kwds=None, **kwargs):
        self.data = model.data
        tsbase.TimeSeriesModelResults.__init__(self, model, params, normalized_cov_params=None, scale=1.0)
        self.filter_results = results
        if isinstance(results, KimSmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type
        self._cache = {}
        if cov_kwds is None:
            cov_kwds = {}
        self._cov_approx_complex_step = cov_kwds.pop('approx_complex_step', True)
        self._cov_approx_centered = cov_kwds.pop('approx_centered', False)
        try:
            self._rank = None
            self._get_robustcov_results(cov_type=cov_type, use_self=True, **cov_kwds)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = 'Covariance matrix could not be calculated: singular. information matrix.'
        attributes = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'predicted_marginal_probabilities', 'predicted_joint_probabilities', 'filtered_marginal_probabilities', 'filtered_joint_probabilities', 'joint_loglikelihoods', 'expected_durations']
        for name in attributes:
            setattr(self, name, getattr(self.filter_results, name))
        attributes = ['smoothed_joint_probabilities', 'smoothed_marginal_probabilities']
        for name in attributes:
            if self.smoother_results is not None:
                setattr(self, name, getattr(self.smoother_results, name))
            else:
                setattr(self, name, None)
        self.predicted_marginal_probabilities = self.predicted_marginal_probabilities.T
        self.filtered_marginal_probabilities = self.filtered_marginal_probabilities.T
        if self.smoother_results is not None:
            self.smoothed_marginal_probabilities = self.smoothed_marginal_probabilities.T
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            if self.expected_durations.ndim > 1:
                self.expected_durations = pd.DataFrame(self.expected_durations, index=index)
            self.predicted_marginal_probabilities = pd.DataFrame(self.predicted_marginal_probabilities, index=index)
            self.filtered_marginal_probabilities = pd.DataFrame(self.filtered_marginal_probabilities, index=index)
            if self.smoother_results is not None:
                self.smoothed_marginal_probabilities = pd.DataFrame(self.smoothed_marginal_probabilities, index=index)

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        pass

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        pass

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        pass

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        pass

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
        pass

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        pass

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        pass

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        pass

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        pass

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        pass

    def predict(self, start=None, end=None, probabilities=None, conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        probabilities : str or array_like, optional
            Specifies the weighting probabilities used in constructing the
            prediction as a weighted average. If a string, can be 'predicted',
            'filtered', or 'smoothed'. Otherwise can be an array of
            probabilities to use. Default is smoothed.
        conditional : bool or int, optional
            Whether or not to return predictions conditional on current or
            past regimes. If False, returns a single vector of weighted
            predictions. If True or 1, returns predictions conditional on the
            current regime. For larger integers, returns predictions
            conditional on the current regime and some number of past regimes.

        Returns
        -------
        predict : ndarray
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        pass

    def forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        pass

    def summary(self, alpha=0.05, start=None, title=None, model_name=None, display_params=True):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title of the summary table.
        model_name : str
            The name of the model used. Default is to use model class name.
        display_params : bool, optional
            Whether or not to display tables of estimated parameters. Default
            is True. Usually only used internally.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        pass

class MarkovSwitchingResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'cov_params_approx': 'cov', 'cov_params_default': 'cov', 'cov_params_opg': 'cov', 'cov_params_robust': 'cov'}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'forecast': 'dates'}
    _wrap_methods = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MarkovSwitchingResultsWrapper, MarkovSwitchingResults)