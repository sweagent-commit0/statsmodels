import numpy as np

class Pca:
    """
    A basic class for Principal Component Analysis (PCA).

    p is the number of dimensions, while N is the number of data points
    """
    _colors = ('r', 'g', 'b', 'c', 'y', 'm', 'k')

    def __init__(self, data, names=None):
        """
        p X N matrix input
        """
        A = np.array(data).T
        n, p = A.shape
        self.n, self.p = (n, p)
        if p > n:
            from warnings import warn
            warn('p > n - intentional?', RuntimeWarning)
        self.A = A
        self._origA = A.copy()
        self.__calc()
        self._colors = np.tile(self._colors, int((p - 1) / len(self._colors)) + 1)[:p]
        if names is not None and len(names) != p:
            raise ValueError('names must match data dimension')
        self.names = None if names is None else tuple([str(x) for x in names])

    def getCovarianceMatrix(self):
        """
        returns the covariance matrix for the dataset
        """
        pass

    def getEigensystem(self):
        """
        returns a tuple of (eigenvalues,eigenvectors) for the data set.
        """
        pass

    def getEnergies(self):
        """
        "energies" are just normalized eigenvectors
        """
        pass

    def plot2d(self, ix=0, iy=1, clf=True):
        """
        Generates a 2-dimensional plot of the data set and principle components
        using matplotlib.

        ix specifies which p-dimension to put on the x-axis of the plot
        and iy specifies which to put on the y-axis (0-indexed)
        """
        pass

    def plot3d(self, ix=0, iy=1, iz=2, clf=True):
        """
        Generates a 3-dimensional plot of the data set and principle components
        using mayavi.

        ix, iy, and iz specify which of the input p-dimensions to place on each of
        the x,y,z axes, respectively (0-indexed).
        """
        pass

    def sigclip(self, sigs):
        """
        clips out all data points that are more than a certain number
        of standard deviations from the mean.

        sigs can be either a single value or a length-p sequence that
        specifies the number of standard deviations along each of the
        p dimensions.
        """
        pass

    def project(self, vals=None, enthresh=None, nPCs=None, cumen=None):
        """
        projects the normalized values onto the components

        enthresh, nPCs, and cumen determine how many PCs to use

        if vals is None, the normalized data vectors are the values to project.
        Otherwise, it should be convertable to a p x N array

        returns n,p(>threshold) dimension array
        """
        pass

    def deproject(self, A, normed=True):
        """
        input is an n X q array, where q <= p

        output is p X n
        """
        pass

    def subtractPC(self, pc, vals=None):
        """
        pc can be a scalar or any sequence of pc indecies

        if vals is None, the source data is self.A, else whatever is in vals
        (which must be p x m)
        """
        pass