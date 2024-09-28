"""Principal Component Analysis


Created on Tue Sep 29 20:11:23 2009
Author: josef-pktd

TODO : add class for better reuse of results
"""
import numpy as np

def pca(data, keepdim=0, normalize=0, demean=True):
    """principal components with eigenvector decomposition
    similar to princomp in matlab

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : int
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    normalize : bool
        if true, then eigenvectors are normalized by sqrt of eigenvalues
    demean : bool
        if true, then the column mean is subtracted from the data

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    Notes
    -----

    See Also
    --------
    pcasvd : principal component analysis using svd

    """
    pass

def pcasvd(data, keepdim=0, demean=True):
    """principal components with svd

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : int
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    demean : bool
        if true, then the column mean is subtracted from the data

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    See Also
    --------
    pca : principal component analysis using eigenvector decomposition

    Notes
    -----
    This does not have yet the normalize option of pca.

    """
    pass
__all__ = ['pca', 'pcasvd']