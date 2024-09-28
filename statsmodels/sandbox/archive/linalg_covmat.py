import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
sqrt2pi = math.sqrt(2 * np.pi)
logsqrt2pi = math.log(sqrt2pi)

class StandardNormal:
    """Distribution of vector x, with independent distribution N(0,1)

    this is the same as univariate normal for pdf and logpdf

    other methods not checked/adjusted yet

    """

class AffineTransform:
    """affine full rank transformation of a multivariate distribution

    no dimension checking, assumes everything broadcasts correctly
    first version without bound support

    provides distribution of y given distribution of x
    y = const + tmat * x

    """

    def __init__(self, const, tmat, dist):
        self.const = const
        self.tmat = tmat
        self.dist = dist
        self.nrv = len(const)
        if not np.equal(self.nrv, tmat.shape).all():
            raise ValueError('dimension of const and tmat do not agree')
        self.tmatinv = linalg.inv(tmat)
        self.absdet = np.abs(np.linalg.det(self.tmat))
        self.logabsdet = np.log(np.abs(np.linalg.det(self.tmat)))
        self.dist

class MultivariateNormalChol:
    """multivariate normal distribution with cholesky decomposition of sigma

    ignoring mean at the beginning, maybe

    needs testing for broadcasting to contemporaneously but not intertemporaly
    correlated random variable, which axis?,
    maybe swapaxis or rollaxis if x.ndim != mean.ndim == (sigma.ndim - 1)

    initially 1d is ok, 2d should work with iid in axis 0 and mvn in axis 1

    """

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.sigmainv = sigmainv
        self.cholsigma = linalg.cholesky(sigma)
        self.cholsigmainv = linalg.cholesky(sigmainv)[::-1, ::-1]

class MultivariateNormal:

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = SvdArray(sigma)

def loglike_ar1(x, rho):
    """loglikelihood of AR(1) process, as a test case

    sigma_u partially hard coded

    Greene chapter 12 eq. (12-31)
    """
    pass

def ar2transform(x, arcoefs):
    """

    (Greene eq 12-30)
    """
    pass

def mvn_loglike(x, sigma):
    """loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    """
    pass

def mvn_nloglike_obs(x, sigma):
    """loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    """
    pass
nobs = 10
x = np.arange(nobs)
autocov = 2 * 0.8 ** np.arange(nobs)
sigma = linalg.toeplitz(autocov)
cholsigma = linalg.cholesky(sigma).T
sigmainv = linalg.inv(sigma)
cholsigmainv = linalg.cholesky(sigmainv)
x_whitened = np.dot(cholsigmainv, x)
logdetsigma = np.log(np.linalg.det(sigma))
sigma2 = 1.0
llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
ll, ls = mvn_nloglike_obs(x, sigma)
print(ll.sum(), 'll.sum()')
print(llike.sum(), 'llike.sum()')
print(np.log(stats.norm._pdf(x_whitened)).sum() - 0.5 * logdetsigma)
print('stats whitened')
print(np.log(stats.norm.pdf(x, scale=np.sqrt(np.diag(sigma)))).sum())
print('stats scaled')
print(0.5 * (np.dot(linalg.cho_solve((linalg.cho_factor(sigma, lower=False)[0].T, False), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5 * (np.dot(linalg.cho_solve((linalg.cho_factor(sigma)[0].T, False), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5 * (np.dot(linalg.cho_solve(linalg.cho_factor(sigma), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(mvn_loglike(x, sigma))
normtransf = AffineTransform(np.zeros(nobs), cholsigma, StandardNormal())
print(normtransf.logpdf(x_whitened).sum())
print(loglike_ar1(x, 0.8))
mch = MultivariateNormalChol(np.zeros(nobs), sigma)
print(mch.logpdf(x))
xw = mch.whiten(x)
print('xSigmax', np.dot(xw, xw))
print('xSigmax', np.dot(x, linalg.cho_solve(linalg.cho_factor(mch.sigma), x)))
print('xSigmax', np.dot(x, linalg.cho_solve((mch.cholsigma, False), x)))