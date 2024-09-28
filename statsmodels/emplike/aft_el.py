"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Stute, W. (1993). "Consistent Estimation Under Random Censorship when
Covariables are Present." Journal of Multivariate Analysis.
Vol. 45. Iss. 1. 89-103

EL and AFT References
---------------------

Zhou, Kim And Bathke. "Empirical Likelihood Analysis for the Heteroskedastic
Accelerated Failure Time Model." Manuscript:
URL: www.ms.uky.edu/~mai/research/CasewiseEL20080724.pdf

Zhou, M. (2005). Empirical Likelihood Ratio with Arbitrarily Censored/
Truncated Data by EM Algorithm.  Journal of Computational and Graphical
Statistics. 14:3, 643-656.


"""
import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts

class OptAFT(_OptFuncts):
    """
    Provides optimization functions used in estimating and conducting
    inference in an AFT model.

    Methods
    ------

    _opt_wtd_nuis_regress:
        Function optimized over nuisance parameters to compute
        the profile likelihood

    _EM_test:
        Uses the modified Em algorithm of Zhou 2005 to maximize the
        likelihood of a parameter vector.
    """

    def __init__(self):
        pass

    def _opt_wtd_nuis_regress(self, test_vals):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------

        params: 1d array
            The regression coefficients of the model.  This includes the
            nuisance and parameters of interests.

        Returns
        -------
        llr : float
            -2 times the log likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
        pass

    def _EM_test(self, nuisance_params, params=None, param_nums=None, b0_vals=None, F=None, survidx=None, uncens_nobs=None, numcensbelow=None, km=None, uncensored=None, censored=None, maxiter=None, ftol=None):
        """
        Uses EM algorithm to compute the maximum likelihood of a test

        Parameters
        ----------

        nuisance_params : ndarray
            Vector of values to be used as nuisance params.

        maxiter : int
            Number of iterations in the EM algorithm for a parameter vector

        Returns
        -------
        -2 ''*'' log likelihood ratio at hypothesized values and
        nuisance params

        Notes
        -----
        Optional parameters are provided by the test_beta function.
        """
        pass

    def _ci_limits_beta(self, b0, param_num=None):
        """
        Returns the difference between the log likelihood for a
        parameter and some critical value.

        Parameters
        ----------
        b0: float
            Value of a regression parameter
        param_num : int
            Parameter index of b0
        """
        pass

class emplikeAFT:
    """

    Class for estimating and conducting inference in an AFT model.

    Parameters
    ----------

    endog: nx1 array
        Response variables that are subject to random censoring

    exog: nxk array
        Matrix of covariates

    censors: nx1 array
        array with entries 0 or 1.  0 indicates a response was
        censored.

    Attributes
    ----------
    nobs : float
        Number of observations
    endog : ndarray
        Endog attay
    exog : ndarray
        Exogenous variable matrix
    censors
        Censors array but sets the max(endog) to uncensored
    nvar : float
        Number of exogenous variables
    uncens_nobs : float
        Number of uncensored observations
    uncens_endog : ndarray
        Uncensored response variables
    uncens_exog : ndarray
        Exogenous variables of the uncensored observations

    Methods
    -------

    params:
        Fits model parameters

    test_beta:
        Tests if beta = b0 for any vector b0.

    Notes
    -----

    The data is immediately sorted in order of increasing endogenous
    variables

    The last observation is assumed to be uncensored which makes
    estimation and inference possible.
    """

    def __init__(self, endog, exog, censors):
        self.nobs = np.shape(exog)[0]
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = np.asarray(censors).reshape(self.nobs, 1)
        self.nvar = self.exog.shape[1]
        idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[idx]
        self.exog = self.exog[idx]
        self.censors = self.censors[idx]
        self.censors[-1] = 1
        self.uncens_nobs = int(np.sum(self.censors))
        mask = self.censors.ravel().astype(bool)
        self.uncens_endog = self.endog[mask, :].reshape(-1, 1)
        self.uncens_exog = self.exog[mask, :]

    def _is_tied(self, endog, censors):
        """
        Indicated if an observation takes the same value as the next
        ordered observation.

        Parameters
        ----------
        endog : ndarray
            Models endogenous variable
        censors : ndarray
            arrat indicating a censored array

        Returns
        -------
        indic_ties : ndarray
            ties[i]=1 if endog[i]==endog[i+1] and
            censors[i]=censors[i+1]
        """
        pass

    def _km_w_ties(self, tie_indic, untied_km):
        """
        Computes KM estimator value at each observation, taking into acocunt
        ties in the data.

        Parameters
        ----------
        tie_indic: 1d array
            Indicates if the i'th observation is the same as the ith +1
        untied_km: 1d array
            Km estimates at each observation assuming no ties.
        """
        pass

    def _make_km(self, endog, censors):
        """

        Computes the Kaplan-Meier estimate for the weights in the AFT model

        Parameters
        ----------
        endog: nx1 array
            Array of response variables
        censors: nx1 array
            Censor-indicating variable

        Returns
        -------
        Kaplan Meier estimate for each observation

        Notes
        -----

        This function makes calls to _is_tied and km_w_ties to handle ties in
        the data.If a censored observation and an uncensored observation has
        the same value, it is assumed that the uncensored happened first.
        """
        pass

    def fit(self):
        """

        Fits an AFT model and returns results instance

        Parameters
        ----------
        None


        Returns
        -------
        Results instance.

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        pass

class AFTResults(OptAFT):

    def __init__(self, model):
        self.model = model

    def params(self):
        """

        Fits an AFT model and returns parameters.

        Parameters
        ----------
        None


        Returns
        -------
        Fitted params

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        pass

    def test_beta(self, b0_vals, param_nums, ftol=10 ** (-5), maxiter=30, print_weights=1):
        """
        Returns the profile log likelihood for regression parameters
        'param_num' at 'b0_vals.'

        Parameters
        ----------
        b0_vals : list
            The value of parameters to be tested
        param_num : list
            Which parameters to be tested
        maxiter : int, optional
            How many iterations to use in the EM algorithm.  Default is 30
        ftol : float, optional
            The function tolerance for the EM optimization.
            Default is 10''**''-5
        print_weights : bool
            If true, returns the weights tate maximize the profile
            log likelihood. Default is False

        Returns
        -------

        test_results : tuple
            The log-likelihood and p-pvalue of the test.

        Notes
        -----

        The function will warn if the EM reaches the maxiter.  However, when
        optimizing over nuisance parameters, it is possible to reach a
        maximum number of inner iterations for a specific value for the
        nuisance parameters while the resultsof the function are still valid.
        This usually occurs when the optimization over the nuisance parameters
        selects parameter values that yield a log-likihood ratio close to
        infinity.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import numpy as np

        # Test parameter is .05 in one regressor no intercept model
        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, x, cens)
        >>> res=model.test_beta([0], [0])
        >>> res
        (1.4657739632606308, 0.22601365256959183)

        #Test slope is 0 in  model with intercept

        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, sm.add_constant(x), cens)
        >>> res = model.test_beta([0], [1])
        >>> res
        (4.623487775078047, 0.031537049752572731)
        """
        pass

    def ci_beta(self, param_num, beta_high, beta_low, sig=0.05):
        """
        Returns the confidence interval for a regression
        parameter in the AFT model.

        Parameters
        ----------
        param_num : int
            Parameter number of interest
        beta_high : float
            Upper bound for the confidence interval
        beta_low : float
            Lower bound for the confidence interval
        sig : float, optional
            Significance level.  Default is .05

        Notes
        -----
        If the function returns f(a) and f(b) must have different signs,
        consider widening the search area by adjusting beta_low and
        beta_high.

        Also note that this process is computational intensive.  There
        are 4 levels of optimization/solving.  From outer to inner:

        1) Solving so that llr-critical value = 0
        2) maximizing over nuisance parameters
        3) Using  EM at each value of nuisamce parameters
        4) Using the _modified_Newton optimizer at each iteration
           of the EM algorithm.

        Also, for very unlikely nuisance parameters, it is possible for
        the EM algorithm to not converge.  This is not an indicator
        that the solver did not find the correct solution.  It just means
        for a specific iteration of the nuisance parameters, the optimizer
        was unable to converge.

        If the user desires to verify the success of the optimization,
        it is recommended to test the limits using test_beta.
        """
        pass