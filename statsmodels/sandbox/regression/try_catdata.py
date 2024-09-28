"""
Working with categorical data
=============================

use of dummy variables, group statistics, within and between statistics
examples for efficient matrix algebra

dummy versions require that the number of unique groups or categories is not too large
group statistics with scipy.ndimage can handle large number of observations and groups
scipy.ndimage stats is missing count

new: np.bincount can also be used for calculating values per label
"""
from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage

def groupstatsbin(factors, values):
    """uses np.bincount, assumes factors/labels are integers
    """
    pass

def convertlabels(ys, indices=None):
    """convert labels based on multiple variables or string labels to unique
    index labels 0,1,2,...,nk-1 where nk is the number of distinct labels
    """
    pass

def groupsstats_1d(y, x, labelsunique):
    """use ndimage to get fast mean and variance"""
    pass