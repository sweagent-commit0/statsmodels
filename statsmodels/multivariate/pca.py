"""Principal Component Analysis

Author: josef-pktd
Modified by Kevin Sheppard
"""
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning, EstimationWarning
from statsmodels.tools.validation import string_like, array_like, bool_like, float_like, int_like

class PCA:
    """
    Principal Component Analysis

    Parameters
    ----------
    data : array_like
        Variables in columns, observations in rows.
    ncomp : int, optional
        Number of components to return.  If None, returns the as many as the
        smaller of the number of rows or columns in data.
    standardize : bool, optional
        Flag indicating to use standardized data with mean 0 and unit
        variance.  standardized being True implies demean.  Using standardized
        data is equivalent to computing principal components from the
        correlation matrix of data.
    demean : bool, optional
        Flag indicating whether to demean data before computing principal
        components.  demean is ignored if standardize is True. Demeaning data
        but not standardizing is equivalent to computing principal components
        from the covariance matrix of data.
    normalize : bool , optional
        Indicates whether to normalize the factors to have unit inner product.
        If False, the loadings will have unit inner product.
    gls : bool, optional
        Flag indicating to implement a two-step GLS estimator where
        in the first step principal components are used to estimate residuals,
        and then the inverse residual variance is used as a set of weights to
        estimate the final principal components.  Setting gls to True requires
        ncomp to be less then the min of the number of rows or columns.
    weights : ndarray, optional
        Series weights to use after transforming data according to standardize
        or demean when computing the principal components.
    method : str, optional
        Sets the linear algebra routine used to compute eigenvectors:

        * 'svd' uses a singular value decomposition (default).
        * 'eig' uses an eigenvalue decomposition of a quadratic form
        * 'nipals' uses the NIPALS algorithm and can be faster than SVD when
          ncomp is small and nvars is large. See notes about additional changes
          when using NIPALS.
    missing : {str, None}
        Method for missing data.  Choices are:

        * 'drop-row' - drop rows with missing values.
        * 'drop-col' - drop columns with missing values.
        * 'drop-min' - drop either rows or columns, choosing by data retention.
        * 'fill-em' - use EM algorithm to fill missing value.  ncomp should be
          set to the number of factors required.
        * `None` raises if data contains NaN values.
    tol : float, optional
        Tolerance to use when checking for convergence when using NIPALS.
    max_iter : int, optional
        Maximum iterations when using NIPALS.
    tol_em : float
        Tolerance to use when checking for convergence of the EM algorithm.
    max_em_iter : int
        Maximum iterations for the EM algorithm.
    svd_full_matrices : bool, optional
        If the 'svd' method is selected, this flag is used to set the parameter
        'full_matrices' in the singular value decomposition method. Is set to
        False by default.

    Attributes
    ----------
    factors : array or DataFrame
        nobs by ncomp array of principal components (scores)
    scores :  array or DataFrame
        nobs by ncomp array of principal components - identical to factors
    loadings : array or DataFrame
        ncomp by nvar array of principal component loadings for constructing
        the factors
    coeff : array or DataFrame
        nvar by ncomp array of principal component loadings for constructing
        the projections
    projection : array or DataFrame
        nobs by var array containing the projection of the data onto the ncomp
        estimated factors
    rsquare : array or Series
        ncomp array where the element in the ith position is the R-square
        of including the fist i principal components.  Note: values are
        calculated on the transformed data, not the original data
    ic : array or DataFrame
        ncomp by 3 array containing the Bai and Ng (2003) Information
        criteria.  Each column is a different criteria, and each row
        represents the number of included factors.
    eigenvals : array or Series
        nvar array of eigenvalues
    eigenvecs : array or DataFrame
        nvar by nvar array of eigenvectors
    weights : ndarray
        nvar array of weights used to compute the principal components,
        normalized to unit length
    transformed_data : ndarray
        Standardized, demeaned and weighted data used to compute
        principal components and related quantities
    cols : ndarray
        Array of indices indicating columns used in the PCA
    rows : ndarray
        Array of indices indicating rows used in the PCA

    Notes
    -----
    The default options perform principal component analysis on the
    demeaned, unit variance version of data.  Setting standardize to False
    will instead only demean, and setting both standardized and
    demean to False will not alter the data.

    Once the data have been transformed, the following relationships hold when
    the number of components (ncomp) is the same as tne minimum of the number
    of observation or the number of variables.

    .. math:

        X' X = V \\Lambda V'

    .. math:

        F = X V

    .. math:

        X = F V'

    where X is the `data`, F is the array of principal components (`factors`
    or `scores`), and V is the array of eigenvectors (`loadings`) and V' is
    the array of factor coefficients (`coeff`).

    When weights are provided, the principal components are computed from the
    modified data

    .. math:

        \\Omega^{-\\frac{1}{2}} X

    where :math:`\\Omega` is a diagonal matrix composed of the weights. For
    example, when using the GLS version of PCA, the elements of :math:`\\Omega`
    will be the inverse of the variances of the residuals from

    .. math:

        X - F V'

    where the number of factors is less than the rank of X

    References
    ----------
    .. [*] J. Bai and S. Ng, "Determining the number of factors in approximate
       factor models," Econometrica, vol. 70, number 1, pp. 191-221, 2002

    Examples
    --------
    Basic PCA using the correlation matrix of the data

    >>> import numpy as np
    >>> from statsmodels.multivariate.pca import PCA
    >>> x = np.random.randn(100)[:, None]
    >>> x = x + np.random.randn(100, 100)
    >>> pc = PCA(x)

    Note that the principal components are computed using a SVD and so the
    correlation matrix is never constructed, unless method='eig'.

    PCA using the covariance matrix of the data

    >>> pc = PCA(x, standardize=False)

    Limiting the number of factors returned to 1 computed using NIPALS

    >>> pc = PCA(x, ncomp=1, method='nipals')
    >>> pc.factors.shape
    (100, 1)
    """

    def __init__(self, data, ncomp=None, standardize=True, demean=True, normalize=True, gls=False, weights=None, method='svd', missing=None, tol=5e-08, max_iter=1000, tol_em=5e-08, max_em_iter=100, svd_full_matrices=False):
        self._index = None
        self._columns = []
        if isinstance(data, pd.DataFrame):
            self._index = data.index
            self._columns = data.columns
        self.data = array_like(data, 'data', ndim=2)
        self._gls = bool_like(gls, 'gls')
        self._normalize = bool_like(normalize, 'normalize')
        self._svd_full_matrices = bool_like(svd_full_matrices, 'svd_fm')
        self._tol = float_like(tol, 'tol')
        if not 0 < self._tol < 1:
            raise ValueError('tol must be strictly between 0 and 1')
        self._max_iter = int_like(max_iter, 'int_like')
        self._max_em_iter = int_like(max_em_iter, 'max_em_iter')
        self._tol_em = float_like(tol_em, 'tol_em')
        self._standardize = bool_like(standardize, 'standardize')
        self._demean = bool_like(demean, 'demean')
        self._nobs, self._nvar = self.data.shape
        weights = array_like(weights, 'weights', maxdim=1, optional=True)
        if weights is None:
            weights = np.ones(self._nvar)
        else:
            weights = np.array(weights).flatten()
            if weights.shape[0] != self._nvar:
                raise ValueError('weights should have nvar elements')
            weights = weights / np.sqrt((weights ** 2.0).mean())
        self.weights = weights
        min_dim = min(self._nobs, self._nvar)
        self._ncomp = min_dim if ncomp is None else ncomp
        if self._ncomp > min_dim:
            import warnings
            warn = 'The requested number of components is more than can be computed from data. The maximum number of components is the minimum of the number of observations or variables'
            warnings.warn(warn, ValueWarning)
            self._ncomp = min_dim
        self._method = method
        if self._method not in ('eig', 'svd', 'nipals'):
            raise ValueError('method {0} is not known.'.format(method))
        if self._method == 'svd':
            self._svd_full_matrices = True
        self.rows = np.arange(self._nobs)
        self.cols = np.arange(self._nvar)
        self._missing = string_like(missing, 'missing', optional=True)
        self._adjusted_data = self.data
        self._adjust_missing()
        self._nobs, self._nvar = self._adjusted_data.shape
        if self._ncomp == np.min(self.data.shape):
            self._ncomp = np.min(self._adjusted_data.shape)
        elif self._ncomp > np.min(self._adjusted_data.shape):
            raise ValueError('When adjusting for missing values, user provided ncomp must be no larger than the smallest dimension of the missing-value-adjusted data size.')
        self._tss = 0.0
        self._ess = None
        self.transformed_data = None
        self._mu = None
        self._sigma = None
        self._ess_indiv = None
        self._tss_indiv = None
        self.scores = self.factors = None
        self.loadings = None
        self.coeff = None
        self.eigenvals = None
        self.eigenvecs = None
        self.projection = None
        self.rsquare = None
        self.ic = None
        self.transformed_data = self._prepare_data()
        self._pca()
        if gls:
            self._compute_gls_weights()
            self.transformed_data = self._prepare_data()
            self._pca()
        self._compute_rsquare_and_ic()
        if self._index is not None:
            self._to_pandas()

    def _adjust_missing(self):
        """
        Implements alternatives for handling missing values
        """
        pass

    def _compute_gls_weights(self):
        """
        Computes GLS weights based on percentage of data fit
        """
        pass

    def _pca(self):
        """
        Main PCA routine
        """
        pass

    def __repr__(self):
        string = self.__str__()
        string = string[:-1]
        string += ', id: ' + hex(id(self)) + ')'
        return string

    def __str__(self):
        string = 'Principal Component Analysis('
        string += 'nobs: ' + str(self._nobs) + ', '
        string += 'nvar: ' + str(self._nvar) + ', '
        if self._standardize:
            kind = 'Standardize (Correlation)'
        elif self._demean:
            kind = 'Demean (Covariance)'
        else:
            kind = 'None'
        string += 'transformation: ' + kind + ', '
        if self._gls:
            string += 'GLS, '
        string += 'normalization: ' + str(self._normalize) + ', '
        string += 'number of components: ' + str(self._ncomp) + ', '
        string += 'method: ' + 'Eigenvalue' if self._method == 'eig' else 'SVD'
        string += ')'
        return string

    def _prepare_data(self):
        """
        Standardize or demean data.
        """
        pass

    def _compute_eig(self):
        """
        Wrapper for actual eigenvalue method

        This is a workaround to avoid instance methods in __dict__
        """
        pass

    def _compute_using_svd(self):
        """SVD method to compute eigenvalues and eigenvecs"""
        pass

    def _compute_using_eig(self):
        """
        Eigenvalue decomposition method to compute eigenvalues and eigenvectors
        """
        pass

    def _compute_using_nipals(self):
        """
        NIPALS implementation to compute small number of eigenvalues
        and eigenvectors
        """
        pass

    def _fill_missing_em(self):
        """
        EM algorithm to fill missing values
        """
        pass

    def _compute_pca_from_eig(self):
        """
        Compute relevant statistics after eigenvalues have been computed
        """
        pass

    def _compute_rsquare_and_ic(self):
        """
        Final statistics to compute
        """
        pass

    def project(self, ncomp=None, transform=True, unweight=True):
        """
        Project series onto a specific number of factors.

        Parameters
        ----------
        ncomp : int, optional
            Number of components to use.  If omitted, all components
            initially computed are used.
        transform : bool, optional
            Flag indicating whether to return the projection in the original
            space of the data (True, default) or in the space of the
            standardized/demeaned data.
        unweight : bool, optional
            Flag indicating whether to undo the effects of the estimation
            weights.

        Returns
        -------
        array_like
            The nobs by nvar array of the projection onto ncomp factors.

        Notes
        -----
        """
        pass

    def _to_pandas(self):
        """
        Returns pandas DataFrames for all values
        """
        pass

    def plot_scree(self, ncomp=None, log_scale=True, cumulative=False, ax=None):
        """
        Plot of the ordered eigenvalues

        Parameters
        ----------
        ncomp : int, optional
            Number of components ot include in the plot.  If None, will
            included the same as the number of components computed
        log_scale : boot, optional
            Flag indicating whether ot use a log scale for the y-axis
        cumulative : bool, optional
            Flag indicating whether to plot the eigenvalues or cumulative
            eigenvalues
        ax : AxesSubplot, optional
            An axes on which to draw the graph.  If omitted, new a figure
            is created

        Returns
        -------
        matplotlib.figure.Figure
            The handle to the figure.
        """
        pass

    def plot_rsquare(self, ncomp=None, ax=None):
        """
        Box plots of the individual series R-square against the number of PCs.

        Parameters
        ----------
        ncomp : int, optional
            Number of components ot include in the plot.  If None, will
            plot the minimum of 10 or the number of computed components.
        ax : AxesSubplot, optional
            An axes on which to draw the graph.  If omitted, new a figure
            is created.

        Returns
        -------
        matplotlib.figure.Figure
            The handle to the figure.
        """
        pass

def pca(data, ncomp=None, standardize=True, demean=True, normalize=True, gls=False, weights=None, method='svd'):
    """
    Perform Principal Component Analysis (PCA).

    Parameters
    ----------
    data : ndarray
        Variables in columns, observations in rows.
    ncomp : int, optional
        Number of components to return.  If None, returns the as many as the
        smaller to the number of rows or columns of data.
    standardize : bool, optional
        Flag indicating to use standardized data with mean 0 and unit
        variance.  standardized being True implies demean.
    demean : bool, optional
        Flag indicating whether to demean data before computing principal
        components.  demean is ignored if standardize is True.
    normalize : bool , optional
        Indicates whether th normalize the factors to have unit inner
        product.  If False, the loadings will have unit inner product.
    gls : bool, optional
        Flag indicating to implement a two-step GLS estimator where
        in the first step principal components are used to estimate residuals,
        and then the inverse residual variance is used as a set of weights to
        estimate the final principal components
    weights : ndarray, optional
        Series weights to use after transforming data according to standardize
        or demean when computing the principal components.
    method : str, optional
        Determines the linear algebra routine uses.  'eig', the default,
        uses an eigenvalue decomposition. 'svd' uses a singular value
        decomposition.

    Returns
    -------
    factors : {ndarray, DataFrame}
        Array (nobs, ncomp) of principal components (also known as scores).
    loadings : {ndarray, DataFrame}
        Array (ncomp, nvar) of principal component loadings for constructing
        the factors.
    projection : {ndarray, DataFrame}
        Array (nobs, nvar) containing the projection of the data onto the ncomp
        estimated factors.
    rsquare : {ndarray, Series}
        Array (ncomp,) where the element in the ith position is the R-square
        of including the fist i principal components.  The values are
        calculated on the transformed data, not the original data.
    ic : {ndarray, DataFrame}
        Array (ncomp, 3) containing the Bai and Ng (2003) Information
        criteria.  Each column is a different criteria, and each row
        represents the number of included factors.
    eigenvals : {ndarray, Series}
        Array of eigenvalues (nvar,).
    eigenvecs : {ndarray, DataFrame}
        Array of eigenvectors. (nvar, nvar).

    Notes
    -----
    This is a simple function wrapper around the PCA class. See PCA for
    more information and additional methods.
    """
    pass