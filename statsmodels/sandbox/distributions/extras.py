"""Various extensions to distributions

* skew normal and skew t distribution by Azzalini, A. & Capitanio, A.
* Gram-Charlier expansion distribution (using 4 moments),
* distributions based on non-linear transformation
  - Transf_gen
  - ExpTransf_gen, LogTransf_gen
  - TransfTwo_gen
    (defines as examples: square, negative square and abs transformations)
  - this versions are without __new__
* mnvormcdf, mvstdnormcdf : cdf, rectangular integral for multivariate normal
  distribution

TODO:
* Where is Transf_gen for general monotonic transformation ? found and added it
* write some docstrings, some parts I do not remember
* add Box-Cox transformation, parametrized ?


this is only partially cleaned, still includes test examples as functions

main changes
* add transf_gen (2010-05-09)
* added separate example and tests (2010-05-09)
* collect transformation function into classes

Example
-------

>>> logtg = Transf_gen(stats.t, np.exp, np.log,
                numargs = 1, a=0, name = 'lnnorm',
                longname = 'Exp transformed normal',
                extradoc = '
distribution of y = exp(x), with x standard normal'
                'precision for moment andstats is not very high, 2-3 decimals')
>>> logtg.cdf(5, 6)
0.92067704211191848
>>> stats.t.cdf(np.log(5), 6)
0.92067704211191848

>>> logtg.pdf(5, 6)
0.021798547904239293
>>> stats.t.pdf(np.log(5), 6)
0.10899273954837908
>>> stats.t.pdf(np.log(5), 6)/5.  #derivative
0.021798547909675815


Author: josef-pktd
License: BSD

"""
import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
try:
    from scipy.stats._mvn import mvndst
except ImportError:
    from scipy.stats.mvn import mvndst

class SkewNorm_gen(distributions.rv_continuous):
    """univariate Skew-Normal distribution of Azzalini

    class follows scipy.stats.distributions pattern
    but with __init__


    """

    def __init__(self):
        distributions.rv_continuous.__init__(self, name='Skew Normal distribution', shapes='alpha')
skewnorm = SkewNorm_gen()

class SkewNorm2_gen(distributions.rv_continuous):
    """univariate Skew-Normal distribution of Azzalini

    class follows scipy.stats.distributions pattern

    """
skewnorm2 = SkewNorm2_gen(name='Skew Normal distribution', shapes='alpha')

class ACSkewT_gen(distributions.rv_continuous):
    """univariate Skew-T distribution of Azzalini

    class follows scipy.stats.distributions pattern
    but with __init__
    """

    def __init__(self):
        distributions.rv_continuous.__init__(self, name='Skew T distribution', shapes='df, alpha')

def pdf_moments_st(cnt):
    """Return the Gaussian expanded pdf function given the list of central
    moments (first one is mean).

    version of scipy.stats, any changes ?
    the scipy.stats version has a bug and returns normal distribution
    """
    pass

def pdf_mvsk(mvsk):
    """Return the Gaussian expanded pdf function given the list of 1st, 2nd
    moment and skew and Fisher (excess) kurtosis.



    Parameters
    ----------
    mvsk : list of mu, mc2, skew, kurt
        distribution is matched to these four moments

    Returns
    -------
    pdffunc : function
        function that evaluates the pdf(x), where x is the non-standardized
        random variable.


    Notes
    -----

    Changed so it works only if four arguments are given. Uses explicit
    formula, not loop.

    This implements a Gram-Charlier expansion of the normal distribution
    where the first 2 moments coincide with those of the normal distribution
    but skew and kurtosis can deviate from it.

    In the Gram-Charlier distribution it is possible that the density
    becomes negative. This is the case when the deviation from the
    normal distribution is too large.



    References
    ----------
    https://en.wikipedia.org/wiki/Edgeworth_series
    Johnson N.L., S. Kotz, N. Balakrishnan: Continuous Univariate
    Distributions, Volume 1, 2nd ed., p.30
    """
    pass

def pdf_moments(cnt):
    """Return the Gaussian expanded pdf function given the list of central
    moments (first one is mean).

    Changed so it works only if four arguments are given. Uses explicit
    formula, not loop.

    Notes
    -----

    This implements a Gram-Charlier expansion of the normal distribution
    where the first 2 moments coincide with those of the normal distribution
    but skew and kurtosis can deviate from it.

    In the Gram-Charlier distribution it is possible that the density
    becomes negative. This is the case when the deviation from the
    normal distribution is too large.



    References
    ----------
    https://en.wikipedia.org/wiki/Edgeworth_series
    Johnson N.L., S. Kotz, N. Balakrishnan: Continuous Univariate
    Distributions, Volume 1, 2nd ed., p.30
    """
    pass

class NormExpan_gen(distributions.rv_continuous):
    """Gram-Charlier Expansion of Normal distribution

    class follows scipy.stats.distributions pattern
    but with __init__

    """

    def __init__(self, args, **kwds):
        distributions.rv_continuous.__init__(self, name='Normal Expansion distribution', shapes=' ')
        mode = kwds.get('mode', 'sample')
        if mode == 'sample':
            mu, sig, sk, kur = stats.describe(args)[2:]
            self.mvsk = (mu, sig, sk, kur)
            cnt = mvsk2mc((mu, sig, sk, kur))
        elif mode == 'mvsk':
            cnt = mvsk2mc(args)
            self.mvsk = args
        elif mode == 'centmom':
            cnt = args
            self.mvsk = mc2mvsk(cnt)
        else:
            raise ValueError("mode must be 'mvsk' or centmom")
        self.cnt = cnt
        self._pdf = pdf_mvsk(self.mvsk)
" A class for the distribution of a non-linear monotonic transformation of a continuous random variable\n\nsimplest usage:\nexample: create log-gamma distribution, i.e. y = log(x),\n            where x is gamma distributed (also available in scipy.stats)\n    loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp)\n\nexample: what is the distribution of the discount factor y=1/(1+x)\n            where interest rate x is normally distributed with N(mux,stdx**2)')?\n            (just to come up with a story that implies a nice transformation)\n    invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf)\n\nThis class does not work well for distributions with difficult shapes,\n    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.\n\nNote: I'm working from my version of scipy.stats.distribution.\n      But this script runs under scipy 0.6.0 (checked with numpy: 1.2.0rc2 and python 2.4)\n\nThis is not yet thoroughly tested, polished or optimized\n\nTODO:\n  * numargs handling is not yet working properly, numargs needs to be specified (default = 0 or 1)\n  * feeding args and kwargs to underlying distribution is untested and incomplete\n  * distinguish args and kwargs for the transformed and the underlying distribution\n    - currently all args and no kwargs are transmitted to underlying distribution\n    - loc and scale only work for transformed, but not for underlying distribution\n    - possible to separate args for transformation and underlying distribution parameters\n\n  * add _rvs as method, will be faster in many cases\n\n\nCreated on Tuesday, October 28, 2008, 12:40:37 PM\nAuthor: josef-pktd\nLicense: BSD\n\n"

class Transf_gen(distributions.rv_continuous):
    """a class for non-linear monotonic transformation of a continuous random variable

    """

    def __init__(self, kls, func, funcinv, *args, **kwargs):
        self.func = func
        self.funcinv = funcinv
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.decr = kwargs.pop('decr', False)
        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls
        super(Transf_gen, self).__init__(a=a, b=b, name=name, longname=longname)
mux, stdx = (0.05, 0.1)
mux, stdx = (9.0, 1.0)
invdnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, numargs=0, name='discf', longname='normal-based discount factor', extradoc='\ndistribution of discount factor y=1/(1+x)) with x N(0.05,0.1**2)')
lognormalg = Transf_gen(stats.norm, np.exp, np.log, numargs=2, a=0, name='lnnorm', longname='Exp transformed normal')
loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp, numargs=1)
'univariate distribution of a non-linear monotonic transformation of a\nrandom variable\n\n'

class ExpTransf_gen(distributions.rv_continuous):
    """Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    """

    def __init__(self, kls, *args, **kwargs):
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(ExpTransf_gen, self).__init__(a=0, name=name)
        self.kls = kls

class LogTransf_gen(distributions.rv_continuous):
    """Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    """

    def __init__(self, kls, *args, **kwargs):
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(LogTransf_gen, self).__init__(a=a, name=name)
        self.kls = kls
'\nCreated on Apr 28, 2009\n\n@author: Josef Perktold\n'
' A class for the distribution of a non-linear u-shaped or hump shaped transformation of a\ncontinuous random variable\n\nThis is a companion to the distributions of non-linear monotonic transformation to the case\nwhen the inverse mapping is a 2-valued correspondence, for example for absolute value or square\n\nsimplest usage:\nexample: create squared distribution, i.e. y = x**2,\n            where x is normal or t distributed\n\n\nThis class does not work well for distributions with difficult shapes,\n    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.\n\n\nThis verifies for normal - chi2, normal - halfnorm, foldnorm, and t - F\n\nTODO:\n  * numargs handling is not yet working properly,\n    numargs needs to be specified (default = 0 or 1)\n  * feeding args and kwargs to underlying distribution works in t distribution example\n  * distinguish args and kwargs for the transformed and the underlying distribution\n    - currently all args and no kwargs are transmitted to underlying distribution\n    - loc and scale only work for transformed, but not for underlying distribution\n    - possible to separate args for transformation and underlying distribution parameters\n\n  * add _rvs as method, will be faster in many cases\n\n'

class TransfTwo_gen(distributions.rv_continuous):
    """Distribution based on a non-monotonic (u- or hump-shaped transformation)

    the constructor can be called with a distribution class, and functions
    that define the non-linear transformation.
    and generates the distribution of the transformed random variable

    Note: the transformation, it's inverse and derivatives need to be fully
    specified: func, funcinvplus, funcinvminus, derivplus,  derivminus.
    Currently no numerical derivatives or inverse are calculated

    This can be used to generate distribution instances similar to the
    distributions in scipy.stats.

    """

    def __init__(self, kls, func, funcinvplus, funcinvminus, derivplus, derivminus, *args, **kwargs):
        self.func = func
        self.funcinvplus = funcinvplus
        self.funcinvminus = funcinvminus
        self.derivplus = derivplus
        self.derivminus = derivminus
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.shape = kwargs.pop('shape', False)
        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls
        super(TransfTwo_gen, self).__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)
        self._ctor_param.update(dict(kls=kls, func=func, funcinvplus=funcinvplus, funcinvminus=funcinvminus, derivplus=derivplus, derivminus=derivminus, shape=self.shape))

class SquareFunc:
    """class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    """
sqfunc = SquareFunc()
squarenormalg = TransfTwo_gen(stats.norm, sqfunc.squarefunc, sqfunc.inverseplus, sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus, shape='u', a=0.0, b=np.inf, numargs=0, name='squarenorm', longname='squared normal distribution')
squaretg = TransfTwo_gen(stats.t, sqfunc.squarefunc, sqfunc.inverseplus, sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus, shape='u', a=0.0, b=np.inf, numargs=1, name='squarenorm', longname='squared t distribution')
negsquarenormalg = TransfTwo_gen(stats.norm, negsquarefunc, inverseplus, inverseminus, derivplus, derivminus, shape='hump', a=-np.inf, b=0.0, numargs=0, name='negsquarenorm', longname='negative squared normal distribution')
absnormalg = TransfTwo_gen(stats.norm, np.abs, inverseplus, inverseminus, derivplus, derivminus, shape='u', a=0.0, b=np.inf, numargs=0, name='absnorm', longname='absolute of normal distribution')
'multivariate normal probabilities and cumulative distribution function\na wrapper for scipy.stats._mvn.mvndst\n\n\n      SUBROUTINE MVNDST( N, LOWER, UPPER, INFIN, CORREL, MAXPTS,\n     &                   ABSEPS, RELEPS, ERROR, VALUE, INFORM )\n*\n*     A subroutine for computing multivariate normal probabilities.\n*     This subroutine uses an algorithm given in the paper\n*     "Numerical Computation of Multivariate Normal Probabilities", in\n*     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by\n*          Alan Genz\n*          Department of Mathematics\n*          Washington State University\n*          Pullman, WA 99164-3113\n*          Email : AlanGenz@wsu.edu\n*\n*  Parameters\n*\n*     N      INTEGER, the number of variables.\n*     LOWER  REAL, array of lower integration limits.\n*     UPPER  REAL, array of upper integration limits.\n*     INFIN  INTEGER, array of integration limits flags:\n*            if INFIN(I) < 0, Ith limits are (-infinity, infinity);\n*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];\n*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);\n*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].\n*     CORREL REAL, array of correlation coefficients; the correlation\n*            coefficient in row I column J of the correlation matrix\n*            should be stored in CORREL( J + ((I-2)*(I-1))/2 ), for J < I.\n*            The correlation matrix must be positive semidefinite.\n*     MAXPTS INTEGER, maximum number of function values allowed. This\n*            parameter can be used to limit the time. A sensible\n*            strategy is to start with MAXPTS = 1000*N, and then\n*            increase MAXPTS if ERROR is too large.\n*     ABSEPS REAL absolute error tolerance.\n*     RELEPS REAL relative error tolerance.\n*     ERROR  REAL estimated absolute error, with 99% confidence level.\n*     VALUE  REAL estimated value for the integral\n*     INFORM INTEGER, termination status parameter:\n*            if INFORM = 0, normal completion with ERROR < EPS;\n*            if INFORM = 1, completion with ERROR > EPS and MAXPTS\n*                           function vaules used; increase MAXPTS to\n*                           decrease ERROR;\n*            if INFORM = 2, N > 500 or N < 1.\n*\n\n\n\n>>> mvndst([0.0,0.0],[10.0,10.0],[0,0],[0.5])\n(2e-016, 1.0, 0)\n>>> mvndst([0.0,0.0],[100.0,100.0],[0,0],[0.0])\n(2e-016, 1.0, 0)\n>>> mvndst([0.0,0.0],[1.0,1.0],[0,0],[0.0])\n(2e-016, 0.70786098173714096, 0)\n>>> mvndst([0.0,0.0],[0.001,1.0],[0,0],[0.0])\n(2e-016, 0.42100802096993045, 0)\n>>> mvndst([0.0,0.0],[0.001,10.0],[0,0],[0.0])\n(2e-016, 0.50039894221391101, 0)\n>>> mvndst([0.0,0.0],[0.001,100.0],[0,0],[0.0])\n(2e-016, 0.50039894221391101, 0)\n>>> mvndst([0.0,0.0],[0.01,100.0],[0,0],[0.0])\n(2e-016, 0.5039893563146316, 0)\n>>> mvndst([0.0,0.0],[0.1,100.0],[0,0],[0.0])\n(2e-016, 0.53982783727702899, 0)\n>>> mvndst([0.0,0.0],[0.1,100.0],[2,2],[0.0])\n(2e-016, 0.019913918638514494, 0)\n>>> mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.0])\n(2e-016, 0.25, 0)\n>>> mvndst([0.0,0.0],[0.0,0.0],[-1,0],[0.0])\n(2e-016, 0.5, 0)\n>>> mvndst([0.0,0.0],[0.0,0.0],[-1,0],[0.5])\n(2e-016, 0.5, 0)\n>>> mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.5])\n(2e-016, 0.33333333333333337, 0)\n>>> mvndst([0.0,0.0],[0.0,0.0],[0,0],[0.99])\n(2e-016, 0.47747329317779391, 0)\n'
informcode = {0: 'normal completion with ERROR < EPS', 1: 'completion with ERROR > EPS and MAXPTS function values used;\n                    increase MAXPTS to decrease ERROR;', 2: 'N > 500 or N < 1'}

def mvstdnormcdf(lower, upper, corrcoef, **kwds):
    """standardized multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats._mvn.mvndst which calculates
    a rectangular integral over a standardized multivariate normal
    distribution.

    This function assumes standardized scale, that is the variance in each dimension
    is one, but correlation can be arbitrary, covariance = correlation matrix

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    corrcoef : float or array_like
       specifies correlation matrix in one of three ways, see notes
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    The correlation matrix corrcoef can be given in 3 different ways
    If the multivariate normal is two-dimensional than only the
    correlation coefficient needs to be provided.
    For general dimension the correlation matrix can be provided either
    as a one-dimensional array of the upper triangular correlation
    coefficients stacked by rows, or as full square correlation matrix

    See Also
    --------
    mvnormcdf : cdf of multivariate normal distribution without
        standardization

    Examples
    --------

    >>> print(mvstdnormcdf([-np.inf,-np.inf], [0.0,np.inf], 0.5))
    0.5
    >>> corr = [[1.0, 0, 0.5],[0,1,0],[0.5,0,1]]
    >>> print(mvstdnormcdf([-np.inf,-np.inf,-100.0], [0.0,0.0,0.0], corr, abseps=1e-6))
    0.166666399198
    >>> print(mvstdnormcdf([-np.inf,-np.inf,-100.0],[0.0,0.0,0.0],corr, abseps=1e-8))
    something wrong completion with ERROR > EPS and MAXPTS function values used;
                        increase MAXPTS to decrease ERROR; 1.048330348e-006
    0.166666546218
    >>> print(mvstdnormcdf([-np.inf,-np.inf,-100.0],[0.0,0.0,0.0], corr,                             maxpts=100000, abseps=1e-8))
    0.166666588293

    """
    pass

def mvnormcdf(upper, mu, cov, lower=None, **kwds):
    """multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats._mvn.mvndst which calculates
    a rectangular integral over a multivariate normal distribution.

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    mu : array_lik, 1d
       list or array of means
    cov : array_like, 2d
       specifies covariance matrix
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    This function normalizes the location and scale of the multivariate
    normal distribution and then uses `mvstdnormcdf` to call the integration.

    See Also
    --------
    mvstdnormcdf : location and scale standardized multivariate normal cdf
    """
    pass