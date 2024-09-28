"""patching scipy to fit distributions and expect method

This adds new methods to estimate continuous distribution parameters with some
fixed/frozen parameters. It also contains functions that calculate the expected
value of a function for any continuous or discrete distribution

It temporarily also contains Bootstrap and Monte Carlo function for testing the
distribution fit, but these are neither general nor verified.

Author: josef-pktd
License: Simplified BSD
"""
from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
stats.distributions.vonmises.a = -np.pi
stats.distributions.vonmises.b = np.pi

def _fitstart(self, x):
    """example method, method of moment estimator as starting values

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    This example was written for the gamma distribution, but not verified
    with literature

    """
    pass

def _fitstart_beta(self, x, fixed=None):
    """method of moment estimator as starting values for beta distribution

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated
    fixed : None or array_like
        sequence of numbers and np.nan to indicate fixed parameters and parameters
        to estimate

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    References
    ----------
    for method of moment estimator for known loc and scale
    https://en.wikipedia.org/wiki/Beta_distribution#Parameter_estimation
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
    NIST reference also includes reference to MLE in
    Johnson, Kotz, and Balakrishan, Volume II, pages 221-235

    """
    pass

def _fitstart_poisson(self, x, fixed=None):
    """maximum likelihood estimator as starting values for Poisson distribution

    Parameters
    ----------
    x : ndarray
        data for which the parameters are estimated
    fixed : None or array_like
        sequence of numbers and np.nan to indicate fixed parameters and parameters
        to estimate

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    References
    ----------
    MLE :
    https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood

    """
    pass

def fit_fr(self, data, *args, **kwds):
    """estimate distribution parameters by MLE taking some parameters as fixed

    Parameters
    ----------
    data : ndarray, 1d
        data for which the distribution parameters are estimated,
    args : list ? check
        starting values for optimization
    kwds :

      - 'frozen' : array_like
           values for frozen distribution parameters and, for elements with
           np.nan, the corresponding parameter will be estimated

    Returns
    -------
    argest : ndarray
        estimated parameters


    Examples
    --------
    generate random sample
    >>> np.random.seed(12345)
    >>> x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)

    estimate all parameters
    >>> stats.gamma.fit(x)
    array([ 2.0243194 ,  0.20395655,  1.44411371])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
    array([ 2.0243194 ,  0.20395655,  1.44411371])

    keep loc fixed, estimate shape and scale parameters
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, np.nan])
    array([ 2.45603985,  1.27333105])

    keep loc and scale fixed, estimate shape parameter
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    array([ 3.00048828])
    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2])
    array([ 2.57792969])

    estimate only scale parameter for fixed shape and loc
    >>> stats.gamma.fit_fr(x, frozen=[2.5, 0.0, np.nan])
    array([ 1.25087891])

    Notes
    -----
    self is an instance of a distribution class. This can be attached to
    scipy.stats.distributions.rv_continuous

    *Todo*

    * check if docstring is correct
    * more input checking, args is list ? might also apply to current fit method

    """
    pass

def expect(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    """calculate expected value of a function with respect to the distribution

    location and scale only tested on a few examples

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution
        conditional : bool (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float

    Notes
    -----
    This function has not been checked for it's behavior when the integral is
    not finite. The integration behavior is inherited from scipy.integrate.quad.

    """
    pass

def expect_v2(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    """calculate expected value of a function with respect to the distribution

    location and scale only tested on a few examples

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set using
           quantiles of the distribution, see Notes
        conditional : bool (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float

    Notes
    -----
    This function has not been checked for it's behavior when the integral is
    not finite. The integration behavior is inherited from scipy.integrate.quad.

    The default limits are lb = self.ppf(1e-9, *args), ub = self.ppf(1-1e-9, *args)

    For some heavy tailed distributions, 'alpha', 'cauchy', 'halfcauchy',
    'levy', 'levy_l', and for 'ncf', the default limits are not set correctly
    even  when the expectation of the function is finite. In this case, the
    integration limits, lb and ub, should be chosen by the user. For example,
    for the ncf distribution, ub=1000 works in the examples.

    There are also problems with numerical integration in some other cases,
    for example if the distribution is very concentrated and the default limits
    are too large.

    """
    pass

def expect_discrete(self, fn=None, args=(), loc=0, lb=None, ub=None, conditional=False):
    """calculate expected value of a function with respect to the distribution
    for discrete distribution

    Parameters
    ----------
        (self : distribution instance as defined in scipy stats)
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        optional keyword parameters
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution, lb and ub are inclusive (ul<=k<=ub)
        conditional : bool (False)
           If true then the expectation is corrected by the conditional
           probability of the integration interval. The return value is the
           expectation of the function, conditional on being in the given
           interval (k such that ul<=k<=ub).

    Returns
    -------
        expected value : float

    Notes
    -----
    * function is not vectorized
    * accuracy: uses self.moment_tol as stopping criterium
        for heavy tailed distribution e.g. zipf(4), accuracy for
        mean, variance in example is only 1e-5,
        increasing precision (moment_tol) makes zipf very slow
    * suppnmin=100 internal parameter for minimum number of points to evaluate
        could be added as keyword parameter, to evaluate functions with
        non-monotonic shapes, points include integers in (-suppnmin, suppnmin)
    * uses maxcount=1000 limits the number of points that are evaluated
        to break loop for infinite sums
        (a maximum of suppnmin+1000 positive plus suppnmin+1000 negative integers
        are evaluated)


    """
    pass
stats.distributions.rv_continuous.fit_fr = fit_fr
stats.distributions.rv_continuous.nnlf_fr = nnlf_fr
stats.distributions.rv_continuous.expect = expect
stats.distributions.rv_discrete.expect = expect_discrete
stats.distributions.beta_gen._fitstart = _fitstart_beta
stats.distributions.poisson_gen._fitstart = _fitstart_poisson

def distfitbootstrap(sample, distr, nrepl=100):
    """run bootstrap for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data for bootstrap
    distr : distribution instance with fit_fr method
    nrepl : int
        number of bootstrap replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all bootstrap replications

    """
    pass

def distfitmc(sample, distr, nrepl=100, distkwds={}):
    """run Monte Carlo for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data, in Monte Carlo only used to get nobs,
    distr : distribution instance with fit_fr method
    nrepl : int
        number of Monte Carlo replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all Monte Carlo replications

    """
    pass

def printresults(sample, arg, bres, kind='bootstrap'):
    """calculate and print(Bootstrap or Monte Carlo result

    Parameters
    ----------
    sample : ndarray
        original sample data
    arg : float   (for general case will be array)
    bres : ndarray
        parameter estimates from Bootstrap or Monte Carlo run
    kind : {'bootstrap', 'montecarlo'}
        output is printed for Mootstrap (default) or Monte Carlo

    Returns
    -------
    None, currently only printing

    Notes
    -----
    still a bit a mess because it is used for both Bootstrap and Monte Carlo

    made correction:
        reference point for bootstrap is estimated parameter

    not clear:
        I'm not doing any ddof adjustment in estimation of variance, do we
        need ddof>0 ?

    todo: return results and string instead of printing

    """
    pass
if __name__ == '__main__':
    examplecases = ['largenumber', 'bootstrap', 'montecarlo'][:]
    if 'largenumber' in examplecases:
        print('\nDistribution: vonmises')
        for nobs in [200]:
            x = stats.vonmises.rvs(1.23, loc=0, scale=1, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('1.23, loc=0, scale=1')
            print('unconstrained')
            print(stats.vonmises.fit(x))
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))
        print('\nDistribution: gamma')
        distr = stats.gamma
        arg, loc, scale = (2.5, 0.0, 20.0)
        for nobs in [200]:
            x = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('%f, loc=%f, scale=%f' % (arg, loc, scale))
            print('unconstrained')
            print(distr.fit(x))
            print(distr.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))
            print('with fixed loc')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, np.nan]))
    ex = ['gamma', 'vonmises'][0]
    if ex == 'gamma':
        distr = stats.gamma
        arg, loc, scale = (2.5, 0.0, 1)
    elif ex == 'vonmises':
        distr = stats.vonmises
        arg, loc, scale = (1.5, 0.0, 1)
    else:
        raise ValueError('wrong example')
    nobs = 100
    nrepl = 1000
    sample = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
    print('\nDistribution:', distr)
    if 'bootstrap' in examplecases:
        print('\nBootstrap')
        bres = distfitbootstrap(sample, distr, nrepl=nrepl)
        printresults(sample, arg, bres)
    if 'montecarlo' in examplecases:
        print('\nMonteCarlo')
        mcres = distfitmc(sample, distr, nrepl=nrepl, distkwds=dict(arg=arg, loc=loc, scale=scale))
        printresults(sample, arg, mcres, kind='montecarlo')