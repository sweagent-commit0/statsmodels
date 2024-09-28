"""
Created on Sun May 10 08:23:48 2015

Author: Josef Perktold
License: BSD-3
"""
import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime

class PenalizedMixin:
    """Mixin class for Maximum Penalized Likelihood

    Parameters
    ----------
    args and kwds for the model super class
    penal : None or instance of Penalized function class
        If penal is None, then NonePenalty is used.
    pen_weight : float or None
        factor for weighting the penalization term.
        If None, then pen_weight is set to nobs.


    TODO: missing **kwds or explicit keywords

    TODO: do we adjust the inherited docstrings?
    We would need templating to add the penalization parameters
    """

    def __init__(self, *args, **kwds):
        self.penal = kwds.pop('penal', None)
        self.pen_weight = kwds.pop('pen_weight', None)
        super(PenalizedMixin, self).__init__(*args, **kwds)
        if self.pen_weight is None:
            self.pen_weight = len(self.endog)
        if self.penal is None:
            self.penal = NonePenalty()
            self.pen_weight = 0
        self._init_keys.extend(['penal', 'pen_weight'])
        self._null_drop_keys = getattr(self, '_null_drop_keys', [])
        self._null_drop_keys.extend(['penal', 'pen_weight'])

    def loglike(self, params, pen_weight=None, **kwds):
        """
        Log-likelihood of model at params
        """
        pass

    def loglikeobs(self, params, pen_weight=None, **kwds):
        """
        Log-likelihood of model observations at params
        """
        pass

    def score_numdiff(self, params, pen_weight=None, method='fd', **kwds):
        """score based on finite difference derivative
        """
        pass

    def score(self, params, pen_weight=None, **kwds):
        """
        Gradient of model at params
        """
        pass

    def score_obs(self, params, pen_weight=None, **kwds):
        """
        Gradient of model observations at params
        """
        pass

    def hessian_numdiff(self, params, pen_weight=None, **kwds):
        """hessian based on finite difference derivative
        """
        pass

    def hessian(self, params, pen_weight=None, **kwds):
        """
        Hessian of model at params
        """
        pass

    def fit(self, method=None, trim=None, **kwds):
        """minimize negative penalized log-likelihood

        Parameters
        ----------
        method : None or str
            Method specifies the scipy optimizer as in nonlinear MLE models.
        trim : {bool, float}
            Default is False or None, which uses no trimming.
            If trim is True or a float, then small parameters are set to zero.
            If True, then a default threshold is used. If trim is a float, then
            it will be used as threshold.
            The default threshold is currently 1e-4, but it will change in
            future and become penalty function dependent.
        kwds : extra keyword arguments
            This keyword arguments are treated in the same way as in the
            fit method of the underlying model class.
            Specifically, additional optimizer keywords and cov_type related
            keywords can be added.
        """
        pass