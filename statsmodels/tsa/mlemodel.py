"""Base Classes for Likelihood Models in time series analysis

Warning: imports numdifftools



Created on Sun Oct 10 15:00:47 2010

Author: josef-pktd
License: BSD

"""
try:
    import numdifftools as ndt
except ImportError:
    pass
from statsmodels.base.model import LikelihoodModel

class TSMLEModel(LikelihoodModel):
    """
    univariate time series model for estimation with maximum likelihood

    Note: This is not working yet
    """

    def __init__(self, endog, exog=None):
        super().__init__(endog, exog)
        self.nar = 1
        self.nma = 1

    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Parameters
        ----------
        params : array_like
            The model parameters

        Notes
        -----
        needs to be overwritten by subclass
        """
        pass

    def score(self, params):
        """
        Score vector for Arma model
        """
        pass

    def hessian(self, params):
        """
        Hessian of arma model.  Currently uses numdifftools
        """
        pass

    def fit(self, start_params=None, maxiter=5000, method='fmin', tol=1e-08):
        """estimate model by minimizing negative loglikelihood

        does this need to be overwritten ?
        """
        pass