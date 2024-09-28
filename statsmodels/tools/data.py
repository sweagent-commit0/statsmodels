"""
Compatibility tools for various data structure inputs
"""
import numpy as np
import pandas as pd

def interpret_data(data, colnames=None, rownames=None):
    """
    Convert passed data structure to form required by estimation classes

    Parameters
    ----------
    data : array_like
    colnames : sequence or None
        May be part of data structure
    rownames : sequence or None

    Returns
    -------
    (values, colnames, rownames) : (homogeneous ndarray, list)
    """
    pass

def _is_recarray(data):
    """
    Returns true if data is a recarray
    """
    pass