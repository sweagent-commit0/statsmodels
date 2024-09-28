"""
Cross-validation classes for GAM

Author: Luca Puggini

"""
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import GenericSmoothers, UnivariateGenericSmoother

class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one
    (e.g. GamCV, LassoCV,...)
    """

    def __init__(self, cv_iterator, endog, exog):
        self.cv_iterator = cv_iterator
        self.exog = exog
        self.endog = endog
        self.train_test_cv_indices = self.cv_iterator.split(self.exog, self.endog, label=None)

def _split_train_test_smoothers(x, smoother, train_index, test_index):
    """split smoothers in test and train sets and create GenericSmoothers

    Note: this does not take exog_linear into account
    """
    pass

class MultivariateGAMCV(BaseCV):

    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.gam = gam
        self.smoother = smoother
        self.exog_linear = exog
        self.alphas = alphas
        self.cv_iterator = cv_iterator
        super(MultivariateGAMCV, self).__init__(cv_iterator, endog, self.smoother.basis)

class BasePenaltiesPathCV(with_metaclass(ABCMeta)):
    """
    Base class for cross validation over a grid of parameters.

    The best parameter is saved in alpha_cv

    This class is currently not used
    """

    def __init__(self, alphas):
        self.alphas = alphas
        self.alpha_cv = None
        self.cv_error = None
        self.cv_std = None

class MultivariateGAMCVPath:
    """k-fold cross-validation for GAM

    Warning: The API of this class is preliminary and will change.

    Parameters
    ----------
    smoother : additive smoother instance
    alphas : list of iteratables
        list of alpha for smooths. The product space will be used as alpha
        grid for cross-validation
    gam : model class
        model class for creating a model with k-fole training data
    cost : function
        cost function for the prediction error
    endog : ndarray
        dependent (response) variable of the model
    cv_iterator : instance of cross-validation iterator
    """

    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.smoother = smoother
        self.gam = gam
        self.alphas = alphas
        self.alphas_grid = list(itertools.product(*self.alphas))
        self.endog = endog
        self.exog = exog
        self.cv_iterator = cv_iterator
        self.cv_error = np.zeros(shape=len(self.alphas_grid))
        self.cv_std = np.zeros(shape=len(self.alphas_grid))
        self.alpha_cv = None