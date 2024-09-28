import numpy as np

def forrt(X, m=None):
    """
    RFFT with order like Munro (1976) FORTT routine.
    """
    pass

def revrt(X, m=None):
    """
    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.
    """
    pass

def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    pass

def counts(x, v):
    """
    Counts the number of elements of x that fall within the grid points v

    Notes
    -----
    Using np.digitize and np.bincount
    """
    pass