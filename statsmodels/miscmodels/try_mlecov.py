"""Multivariate Normal Model with full covariance matrix

toeplitz structure is not exploited, need cholesky or inv for toeplitz

Author: josef-pktd
"""
import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf, arma_generate_sample

def mvn_loglike_sum(x, sigma):
    """loglike multivariate normal

    copied from GLS and adjusted names
    not sure why this differes from mvn_loglike
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

def mvn_loglike_chol(x, sigma):
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

class MLEGLS(GenericLikelihoodModel):
    """ARMA model with exact loglikelhood for short time series

    Inverts (nobs, nobs) matrix, use only for nobs <= 200 or so.

    This class is a pattern for small sample GLS-like models. Intended use
    for loglikelihood of initial observations for ARMA.



    TODO:
    This might be missing the error variance. Does it assume error is
       distributed N(0,1)
    Maybe extend to mean handling, or assume it is already removed.
    """

    def _params2cov(self, params, nobs):
        """get autocovariance matrix from ARMA regression parameter

        ar parameters are assumed to have rhs parameterization

        """
        pass
if __name__ == '__main__':
    nobs = 50
    ar = [1.0, -0.8, 0.1]
    ma = [1.0, 0.1, 0.2]
    np.random.seed(9875789)
    y = arma_generate_sample(ar, ma, nobs, 2)
    y -= y.mean()
    mod = MLEGLS(y)
    mod.nar, mod.nma = (2, 2)
    mod.nobs = len(y)
    res = mod.fit(start_params=[0.1, -0.8, 0.2, 0.1, 1.0])
    print('DGP', ar, ma)
    print(res.params)
    from statsmodels.regression import yule_walker
    print(yule_walker(y, 2))
    arpoly, mapoly = getpoly(mod, res.params[:-1])
    data = sunspots.load()
    sigma = mod._params2cov(res.params[:-1], nobs) * res.params[-1] ** 2
    print(mvn_loglike(y, sigma))
    llo = mvn_nloglike_obs(y, sigma)
    print(llo.sum(), llo.shape)
    print(mvn_loglike_chol(y, sigma))
    print(mvn_loglike_sum(y, sigma))