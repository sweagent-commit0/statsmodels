"""
Cross-validation iterators for GAM

Author: Luca Puggini

"""
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np

class BaseCrossValidator(with_metaclass(ABCMeta)):
    """
    The BaseCrossValidator class is a base class for all the iterators that
    split the data in train and test as for example KFolds or LeavePOut
    """

    def __init__(self):
        pass

class KFold(BaseCrossValidator):
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets

    Parameters
    ----------
    k: int
        number of folds
    shuffle : bool
        If true, then the index is shuffled before splitting into train and
        test indices.

    Notes
    -----
    All folds except for last fold have size trunc(n/k), the last fold has
    the remainder.
    """

    def __init__(self, k_folds, shuffle=False):
        self.nobs = None
        self.k_folds = k_folds
        self.shuffle = shuffle

    def split(self, X, y=None, label=None):
        """yield index split into train and test sets
        """
        pass