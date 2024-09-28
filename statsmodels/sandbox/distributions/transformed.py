""" A class for the distribution of a non-linear monotonic transformation of a continuous random variable

simplest usage:
example: create log-gamma distribution, i.e. y = log(x),
            where x is gamma distributed (also available in scipy.stats)
    loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp)

example: what is the distribution of the discount factor y=1/(1+x)
            where interest rate x is normally distributed with N(mux,stdx**2)')?
            (just to come up with a story that implies a nice transformation)
    invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf)

This class does not work well for distributions with difficult shapes,
    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.

Note: I'm working from my version of scipy.stats.distribution.
      But this script runs under scipy 0.6.0 (checked with numpy: 1.2.0rc2 and python 2.4)

This is not yet thoroughly tested, polished or optimized

TODO:
  * numargs handling is not yet working properly, numargs needs to be specified (default = 0 or 1)
  * feeding args and kwargs to underlying distribution is untested and incomplete
  * distinguish args and kwargs for the transformed and the underlying distribution
    - currently all args and no kwargs are transmitted to underlying distribution
    - loc and scale only work for transformed, but not for underlying distribution
    - possible to separate args for transformation and underlying distribution parameters

  * add _rvs as method, will be faster in many cases


Created on Tuesday, October 28, 2008, 12:40:37 PM
Author: josef-pktd
License: BSD

"""
from scipy import stats
from scipy.stats import distributions
import numpy as np

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
        super(Transf_gen, self).__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)
mux, stdx = (0.05, 0.1)
mux, stdx = (9.0, 1.0)
invdnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, numargs=0, name='discf', longname='normal-based discount factor')
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
        super(ExpTransf_gen, self).__init__(a=a, name=name)
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