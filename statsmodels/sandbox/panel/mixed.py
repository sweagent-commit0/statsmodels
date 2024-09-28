"""
Mixed effects models

Author: Jonathan Taylor
Author: Josef Perktold
License: BSD-3


Notes
-----

It's pretty slow if the model is misspecified, in my first example convergence
in loglike is not reached within 2000 iterations. Added stop criteria based
on convergence of parameters instead.

With correctly specified model, convergence is fast, in 6 iterations in
example.

"""
import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly

class Unit:
    """
    Individual experimental unit for
    EM implementation of (repeated measures)
    mixed effects model.

    'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    endog : ndarray, (nobs,)
        response, endogenous variable
    exog_fe : ndarray, (nobs, k_vars_fe)
        explanatory variables as regressors or fixed effects,
        should include exog_re to correct mean of random
        coefficients, see Notes
    exog_re : ndarray, (nobs, k_vars_re)
        explanatory variables or random effects or coefficients

    Notes
    -----
    If the exog_re variables are not included in exog_fe, then the
    mean of the random constants or coefficients are not centered.
    The covariance matrix of the random parameter estimates are not
    centered in this case. (That's how it looks to me. JP)
    """

    def __init__(self, endog, exog_fe, exog_re):
        self.Y = endog
        self.X = exog_fe
        self.Z = exog_re
        self.n = endog.shape[0]

    def _compute_S(self, D, sigma):
        """covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.3) from Laird, Lange, Stram (see help(Unit))
        """
        pass

    def _compute_W(self):
        """inverse covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.2) from Laird, Lange, Stram (see help(Unit))
        """
        pass

    def compute_P(self, Sinv):
        """projection matrix (nobs_i, nobs_i) (M in regression ?)  (JP check, guessing)
        Display (3.10) from Laird, Lange, Stram (see help(Unit))

        W - W X Sinv X' W'
        """
        pass

    def _compute_r(self, alpha):
        """residual after removing fixed effects

        Display (3.5) from Laird, Lange, Stram (see help(Unit))
        """
        pass

    def _compute_b(self, D):
        """coefficients for random effects/coefficients
        Display (3.4) from Laird, Lange, Stram (see help(Unit))

        D Z' W r
        """
        pass

    def fit(self, a, D, sigma):
        """
        Compute unit specific parameters in
        Laird, Lange, Stram (see help(Unit)).

        Displays (3.2)-(3.5).
        """
        pass

    def compute_xtwy(self):
        """
        Utility function to compute X^tWY (transposed ?) for Unit instance.
        """
        pass

    def compute_xtwx(self):
        """
        Utility function to compute X^tWX for Unit instance.
        """
        pass

    def cov_random(self, D, Sinv=None):
        """
        Approximate covariance of estimates of random effects. Just after
        Display (3.10) in Laird, Lange, Stram (see help(Unit)).

        D - D' Z' P Z D

        Notes
        -----
        In example where the mean of the random coefficient is not zero, this
        is not a covariance but a non-centered moment. (proof by example)
        """
        pass

    def logL(self, a, ML=False):
        """
        Individual contributions to the log-likelihood, tries to return REML
        contribution by default though this requires estimated
        fixed effect a to be passed as an argument.

        no constant with pi included

        a is not used if ML=true  (should be a=None in signature)
        If ML is false, then the residuals are calculated for the given fixed
        effects parameters a.
        """
        pass

    def deviance(self, ML=False):
        """deviance defined as 2 times the negative loglikelihood

        """
        pass

class OneWayMixed:
    """
    Model for
    EM implementation of (repeated measures)
    mixed effects model.

    'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    units : list of units
       the data for the individual units should be attached to the units
    response, fixed and random : formula expression, called as argument to Formula


    *available results and alias*

    (subject to renaming, and coversion to cached attributes)

    params() -> self.a : coefficient for fixed effects or exog
    cov_params() -> self.Sinv : covariance estimate of fixed effects/exog
    bse() : standard deviation of params

    cov_random -> self.D : estimate of random effects covariance
    params_random_units -> [self.units[...].b] : random coefficient for each unit


    *attributes*

    (others)

    self.m : number of units
    self.p : k_vars_fixed
    self.q : k_vars_random
    self.N : nobs (total)


    Notes
    -----
    Fit returns a result instance, but not all results that use the inherited
    methods have been checked.

    Parameters need to change: drop formula and we require a naming convention for
    the units (currently Y,X,Z). - endog, exog_fe, endog_re ?

    logL does not include constant, e.g. sqrt(pi)
    llf is for MLE not for REML


    convergence criteria for iteration
    Currently convergence in the iterative solver is reached if either the loglikelihood
    *or* the fixed effects parameter do not change above tolerance.

    In some examples, the fixed effects parameters converged to 1e-5 within 150 iterations
    while the log likelihood did not converge within 2000 iterations. This might be
    the case if the fixed effects parameters are well estimated, but there are still
    changes in the random effects. If params_rtol and params_atol are set at a higher
    level, then the random effects might not be estimated to a very high precision.

    The above was with a misspecified model, without a constant. With a
    correctly specified model convergence is fast, within a few iterations
    (6 in example).
    """

    def __init__(self, units):
        self.units = units
        self.m = len(self.units)
        self.n_units = self.m
        self.N = sum((unit.X.shape[0] for unit in self.units))
        self.nobs = self.N
        d = self.units[0].X
        self.p = d.shape[1]
        self.k_exog_fe = self.p
        self.a = np.zeros(self.p, np.float64)
        d = self.units[0].Z
        self.q = d.shape[1]
        self.k_exog_re = self.q
        self.D = np.zeros((self.q,) * 2, np.float64)
        self.sigma = 1.0
        self.dev = np.inf

    def _compute_a(self):
        """fixed effects parameters

        Display (3.1) of
        Laird, Lange, Stram (see help(Mixed)).
        """
        pass

    def _compute_sigma(self, ML=False):
        """
        Estimate sigma. If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.6) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.8).

        sigma is the standard deviation of the noise (residual)
        """
        pass

    def _compute_D(self, ML=False):
        """
        Estimate random effects covariance D.
        If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.7) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.9).
        """
        pass

    def cov_fixed(self):
        """
        Approximate covariance of estimates of fixed effects.

        Just after Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
        pass

    def cov_random(self):
        """
        Estimate random effects covariance D.

        If ML is True, return the ML estimate of sigma, else return the REML estimate.

        see _compute_D, alias for self.D
        """
        pass

    @property
    def params(self):
        """
        estimated coefficients for exogeneous variables or fixed effects

        see _compute_a, alias for self.a
        """
        pass

    @property
    def params_random_units(self):
        """random coefficients for each unit

        """
        pass

    def cov_params(self):
        """
        estimated covariance for coefficients for exogeneous variables or fixed effects

        see cov_fixed, and Sinv in _compute_a
        """
        pass

    @property
    def bse(self):
        """
        standard errors of estimated coefficients for exogeneous variables (fixed)

        """
        pass

    def deviance(self, ML=False):
        """deviance defined as 2 times the negative loglikelihood

        """
        pass

    def logL(self, ML=False):
        """
        Return log-likelihood, REML by default.
        """
        pass

    def cont(self, ML=False, rtol=1e-05, params_rtol=1e-05, params_atol=0.0001):
        """convergence check for iterative estimation

        """
        pass

class OneWayMixedResults(LikelihoodModelResults):
    """Results class for OneWayMixed models

    """

    def __init__(self, model):
        self.model = model
        self.params = model.params

    def plot_random_univariate(self, bins=None, use_loc=True):
        """create plot of marginal distribution of random effects

        Parameters
        ----------
        bins : int or bin edges
            option for bins in matplotlibs hist method. Current default is not
            very sophisticated. All distributions use the same setting for
            bins.
        use_loc : bool
            If True, then the distribution with mean given by the fixed
            effect is used.

        Returns
        -------
        Figure
            figure with subplots

        Notes
        -----
        What can make this fancier?

        Bin edges will not make sense if loc or scale differ across random
        effect distributions.

        """
        pass

    def plot_scatter_pairs(self, idx1, idx2, title=None, ax=None):
        """create scatter plot of two random effects

        Parameters
        ----------
        idx1, idx2 : int
            indices of the two random effects to display, corresponding to
            columns of exog_re
        title : None or string
            If None, then a default title is added
        ax : None or matplotlib axis instance
            If None, then a figure with one axis is created and returned.
            If ax is not None, then the scatter plot is created on it, and
            this axis instance is returned.

        Returns
        -------
        ax_or_fig : axis or figure instance
            see ax parameter

        Notes
        -----
        Still needs ellipse from estimated parameters

        """
        pass