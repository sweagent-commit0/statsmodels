"""
Generalized additive models



Requirements for smoothers
--------------------------

smooth(y, weights=xxx) : ? no return ? alias for fit
predict(x=None) : smoothed values, fittedvalues or for new exog
df_fit() : degress of freedom of fit ?


Notes
-----
- using PolySmoother works for AdditiveModel, and GAM with Poisson and Binomial
- testfailure with Gamma, no other families tested
- there is still an indeterminacy in the split up of the constant across
  components (smoothers) and alpha, sum, i.e. constant, looks good.
  - role of offset, that I have not tried to figure out yet

Refactoring
-----------
currently result is attached to model instead of other way around
split up Result in class for AdditiveModel and for GAM,
subclass GLMResults, needs verification that result statistics are appropriate
how much inheritance, double inheritance?
renamings and cleanup
interface to other smoothers, scipy splines

basic unittests as support for refactoring exist, but we should have a test
case for gamma and the others. Advantage of PolySmoother is that we can
benchmark against the parametric GLM results.

"""
import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
DEBUG = False

def default_smoother(x, s_arg=None):
    """

    """
    pass

class Offset:

    def __init__(self, fn, offset):
        self.fn = fn
        self.offset = offset

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw) + self.offset

class Results:

    def __init__(self, Y, alpha, exog, smoothers, family, offset):
        self.nobs, self.k_vars = exog.shape
        self.Y = Y
        self.alpha = alpha
        self.smoothers = smoothers
        self.offset = offset
        self.family = family
        self.exog = exog
        self.offset = offset
        self.mu = self.linkinversepredict(exog)

    def __call__(self, exog):
        """expected value ? check new GLM, same as mu for given exog
        maybe remove this
        """
        return self.linkinversepredict(exog)

    def linkinversepredict(self, exog):
        """expected value ? check new GLM, same as mu for given exog
        """
        pass

    def predict(self, exog):
        """predict response, sum of smoothed components
        TODO: What's this in the case of GLM, corresponds to X*beta ?
        """
        pass

    def smoothed(self, exog):
        """get smoothed prediction for each component

        """
        pass

class AdditiveModel:
    """additive model with non-parametric, smoothed components

    Parameters
    ----------
    exog : ndarray
    smoothers : None or list of smoother instances
        smoother instances not yet checked
    weights : None or ndarray
    family : None or family instance
        I think only used because of shared results with GAM and subclassing.
        If None, then Gaussian is used.
    """

    def __init__(self, exog, smoothers=None, weights=None, family=None):
        self.exog = exog
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.exog.shape[0])
        self.smoothers = smoothers or [default_smoother(exog[:, i]) for i in range(exog.shape[1])]
        for i in range(exog.shape[1]):
            self.smoothers[i].df = 10
        if family is None:
            self.family = families.Gaussian()
        else:
            self.family = family

    def _iter__(self):
        """initialize iteration ?, should be removed

        """
        pass

    def next(self):
        """internal calculation for one fit iteration

        BUG: I think this does not improve, what is supposed to improve
            offset does not seem to be used, neither an old alpha
            The smoothers keep coef/params from previous iteration
        """
        pass

    def cont(self):
        """condition to continue iteration loop

        Parameters
        ----------
        tol

        Returns
        -------
        cont : bool
            If true, then iteration should be continued.

        """
        pass

    def df_resid(self):
        """degrees of freedom of residuals, ddof is sum of all smoothers df
        """
        pass

    def estimate_scale(self):
        """estimate standard deviation of residuals
        """
        pass

    def fit(self, Y, rtol=1e-06, maxiter=30):
        """fit the model to a given endogenous variable Y

        This needs to change for consistency with statsmodels

        """
        pass

class Model(GLM, AdditiveModel):

    def __init__(self, endog, exog, smoothers=None, family=families.Gaussian()):
        AdditiveModel.__init__(self, exog, smoothers=smoothers, family=family)
        GLM.__init__(self, endog, exog, family=family)
        assert self.family is family

    def estimate_scale(self, Y=None):
        """
        Return Pearson's X^2 estimate of scale.
        """
        pass