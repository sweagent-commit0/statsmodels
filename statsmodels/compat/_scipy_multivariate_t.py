import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.stats import mvn
_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)
_doc_random_state = 'random_state : {None, int, np.random.RandomState, np.random.Generator}, optional\n    Used for drawing random variates.\n    If `seed` is `None` the `~np.random.RandomState` singleton is used.\n    If `seed` is an int, a new ``RandomState`` instance is used, seeded\n    with seed.\n    If `seed` is already a ``RandomState`` or ``Generator`` instance,\n    then that object is used.\n    Default is None.\n'

def _squeeze_output(out):
    """
    Remove single-dimensional entries from array and convert to scalar,
    if necessary.

    """
    pass

def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    pass

def _pinv_1d(v, eps=1e-05):
    """
    A helper function for computing the pseudoinverse.

    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.

    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.

    """
    pass

class _PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.

    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().

    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)

    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().

    """

    def __init__(self, M, cond=None, rcond=None, lower=True, check_finite=True, allow_singular=True):
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)
        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and (not allow_singular):
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))
        self._pinv = None

class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.

    """

    def __init__(self, seed=None):
        super(multi_rv_generic, self).__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the RandomState object for generating random variates.

        This can be either None, int, a RandomState instance, or a
        np.random.Generator instance.

        If None (or np.random), use the RandomState singleton used by
        np.random.
        If already a RandomState or Generator instance, use it.
        If an int, use a new RandomState instance seeded with seed.

        """
        pass

class multi_rv_frozen:
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """
_mvn_doc_default_callparams = 'mean : array_like, optional\n    Mean of the distribution (default zero)\ncov : array_like, optional\n    Covariance matrix of the distribution (default one)\nallow_singular : bool, optional\n    Whether to allow a singular covariance matrix.  (Default: False)\n'
_mvn_doc_callparams_note = 'Setting the parameter `mean` to `None` is equivalent to having `mean`\n    be the zero-vector. The parameter `cov` can be a scalar, in which case\n    the covariance matrix is the identity times that value, a vector of\n    diagonal entries for the covariance matrix, or a two-dimensional\n    array_like.\n    '
_mvn_doc_frozen_callparams = ''
_mvn_doc_frozen_callparams_note = 'See class definition for a detailed description of parameters.'
mvn_docdict_params = {'_mvn_doc_default_callparams': _mvn_doc_default_callparams, '_mvn_doc_callparams_note': _mvn_doc_callparams_note, '_doc_random_state': _doc_random_state}
mvn_docdict_noparams = {'_mvn_doc_default_callparams': _mvn_doc_frozen_callparams, '_mvn_doc_callparams_note': _mvn_doc_frozen_callparams_note, '_doc_random_state': _doc_random_state}

class multivariate_normal_gen(multi_rv_generic):
    """
    A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    ``pdf(x, mean=None, cov=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, mean=None, cov=1, allow_singular=False)``
        Log of the probability density function.
    ``cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Cumulative distribution function.
    ``logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5)``
        Log of the cumulative distribution function.
    ``rvs(mean=None, cov=1, size=1, random_state=None)``
        Draw random samples from a multivariate normal distribution.
    ``entropy()``
        Compute the differential entropy of the multivariate normal.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvn_doc_default_callparams)s
    %(_doc_random_state)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    rv = multivariate_normal(mean=None, cov=1, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    %(_mvn_doc_callparams_note)s

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \\frac{1}{\\sqrt{(2 \\pi)^k \\det \\Sigma}}
               \\exp\\left( -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu) \\right),

    where :math:`\\mu` is the mean, :math:`\\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_normal

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        super(multivariate_normal_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvn_docdict_params)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """
        Create a frozen multivariate normal distribution.

        See `multivariate_normal_frozen` for more information.

        """
        return multivariate_normal_frozen(mean, cov, allow_singular=allow_singular, seed=seed)

    def _process_parameters(self, dim, mean, cov):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.

        """
        pass

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        pass

    def _logpdf(self, x, mean, prec_U, log_det_cov, rank):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix
        rank : int
            Rank of the covariance matrix.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        pass

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        pass

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        """
        Multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        pass

    def _cdf(self, x, mean, cov, maxpts, abseps, releps):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        mean : ndarray
            Mean of the distribution
        cov : array_like
            Covariance matrix of the distribution
        maxpts: integer
            The maximum number of points to use for integration
        abseps: float
            Absolute error tolerance
        releps: float
            Relative error tolerance

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.

        .. versionadded:: 1.0.0

        """
        pass

    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05):
        """
        Log of the multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        pass

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None, abseps=1e-05, releps=1e-05):
        """
        Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts: integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance (default 1e-5)
        releps: float, optional
            Relative error tolerance (default 1e-5)

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        pass

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        pass

    def entropy(self, mean=None, cov=1):
        """
        Compute the differential entropy of the multivariate normal.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        pass
multivariate_normal = multivariate_normal_gen()

class multivariate_normal_frozen(multi_rv_frozen):

    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None, maxpts=None, abseps=1e-05, releps=1e-05):
        """
        Create a frozen multivariate normal distribution.

        Parameters
        ----------
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)
        allow_singular : bool, optional
            If this flag is True then tolerate a singular
            covariance matrix (default False).
        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            This parameter defines the object to use for drawing random
            variates.
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.
        maxpts: integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default `1000000*dim`)
        abseps: float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps: float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_normal
        >>> r = multivariate_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        self._dist = multivariate_normal_gen(seed)
        self.dim, self.mean, self.cov = self._dist._process_parameters(None, mean, cov)
        self.cov_info = _PSD(self.cov, allow_singular=allow_singular)
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    def entropy(self):
        """
        Computes the differential entropy of the multivariate normal.

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        """
        pass
_mvt_doc_default_callparams = '\nloc : array_like, optional\n    Location of the distribution. (default ``0``)\nshape : array_like, optional\n    Positive semidefinite matrix of the distribution. (default ``1``)\ndf : float, optional\n    Degrees of freedom of the distribution; must be greater than zero.\n    If ``np.inf`` then results are multivariate normal. The default is ``1``.\nallow_singular : bool, optional\n    Whether to allow a singular matrix. (default ``False``)\n'
_mvt_doc_callparams_note = 'Setting the parameter `loc` to ``None`` is equivalent to having `loc`\nbe the zero-vector. The parameter `shape` can be a scalar, in which case\nthe shape matrix is the identity times that value, a vector of\ndiagonal entries for the shape matrix, or a two-dimensional array_like.\n'
_mvt_doc_frozen_callparams_note = 'See class definition for a detailed description of parameters.'
mvt_docdict_params = {'_mvt_doc_default_callparams': _mvt_doc_default_callparams, '_mvt_doc_callparams_note': _mvt_doc_callparams_note, '_doc_random_state': _doc_random_state}
mvt_docdict_noparams = {'_mvt_doc_default_callparams': '', '_mvt_doc_callparams_note': _mvt_doc_frozen_callparams_note, '_doc_random_state': _doc_random_state}

class multivariate_t_gen(multi_rv_generic):
    """
    A multivariate t-distributed random variable.

    The `loc` parameter specifies the location. The `shape` parameter specifies
    the positive semidefinite shape matrix. The `df` parameter specifies the
    degrees of freedom.

    In addition to calling the methods below, the object itself may be called
    as a function to fix the location, shape matrix, and degrees of freedom
    parameters, returning a "frozen" multivariate t-distribution random.

    Methods
    -------
    ``pdf(x, loc=None, shape=1, df=1, allow_singular=False)``
        Probability density function.
    ``logpdf(x, loc=None, shape=1, df=1, allow_singular=False)``
        Log of the probability density function.
    ``rvs(loc=None, shape=1, df=1, size=1, random_state=None)``
        Draw random samples from a multivariate t-distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvt_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_mvt_doc_callparams_note)s
    The matrix `shape` must be a (symmetric) positive semidefinite matrix. The
    determinant and inverse of `shape` are computed as the pseudo-determinant
    and pseudo-inverse, respectively, so that `shape` does not need to have
    full rank.

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = \\frac{\\Gamma(\\nu + p)/2}{\\Gamma(\\nu/2)\\nu^{p/2}\\pi^{p/2}|\\Sigma|^{1/2}}
               \\exp\\left[1 + \\frac{1}{\\nu} (\\mathbf{x} - \\boldsymbol{\\mu})^{\\top}
               \\boldsymbol{\\Sigma}^{-1}
               (\\mathbf{x} - \\boldsymbol{\\mu}) \\right]^{-(\\nu + p)/2},

    where :math:`p` is the dimension of :math:`\\mathbf{x}`,
    :math:`\\boldsymbol{\\mu}` is the :math:`p`-dimensional location,
    :math:`\\boldsymbol{\\Sigma}` the :math:`p \\times p`-dimensional shape
    matrix, and :math:`\\nu` is the degrees of freedom.

    .. versionadded:: 1.6.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import multivariate_t
    >>> x, y = np.mgrid[-1:3:.01, -2:1.5:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2)
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.set_aspect('equal')
    >>> plt.contourf(x, y, rv.pdf(pos))

    """

    def __init__(self, seed=None):
        """
        Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.

        """
        super(multivariate_t_gen, self).__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        """
        Create a frozen multivariate t-distribution. See
        `multivariate_t_frozen` for parameters.

        """
        if df == np.inf:
            return multivariate_normal_frozen(mean=loc, cov=shape, allow_singular=allow_singular, seed=seed)
        return multivariate_t_frozen(loc=loc, shape=shape, df=df, allow_singular=allow_singular, seed=seed)

    def pdf(self, x, loc=None, shape=1, df=1, allow_singular=False):
        """
        Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : Probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.pdf(x, loc, shape, df)
        array([0.00075713])

        """
        pass

    def logpdf(self, x, loc=None, shape=1, df=1):
        """
        Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.logpdf(x, loc, shape, df)
        array([-7.1859802])

        See Also
        --------
        pdf : Probability density function.

        """
        pass

    def _logpdf(self, x, loc, prec_U, log_pdet, df, dim, rank):
        """Utility method `pdf`, `logpdf` for parameters.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability density
            function.
        loc : ndarray
            Location of the distribution.
        prec_U : ndarray
            A decomposition such that `np.dot(prec_U, prec_U.T)` is the inverse
            of the shape matrix.
        log_pdet : float
            Logarithm of the determinant of the shape matrix.
        df : float
            Degrees of freedom of the distribution.
        dim : int
            Dimension of the quantiles x.
        rank : int
            Rank of the shape matrix.

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly; use 'logpdf' instead.

        """
        pass

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        """
        Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `P`), where `P` is the
            dimension of the random variable.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.rvs(loc, shape, df)
        array([[0.93477495, 3.00408716]])

        """
        pass

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        pass

    def _process_parameters(self, loc, shape, df):
        """
        Infer dimensionality from location array and shape matrix, handle
        defaults, and ensure compatible dimensions.

        """
        pass

class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        """
        Create a frozen multivariate t distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s

        Examples
        --------
        >>> loc = np.zeros(3)
        >>> shape = np.eye(3)
        >>> df = 10
        >>> dist = multivariate_t(loc, shape, df)
        >>> dist.rvs()
        array([[ 0.81412036, -1.53612361,  0.42199647]])
        >>> dist.pdf([1, 1, 1])
        array([0.01237803])

        """
        self._dist = multivariate_t_gen(seed)
        dim, loc, shape, df = self._dist._process_parameters(loc, shape, df)
        self.dim, self.loc, self.shape, self.df = (dim, loc, shape, df)
        self.shape_info = _PSD(shape, allow_singular=allow_singular)
multivariate_t = multivariate_t_gen()
for name in ['logpdf', 'pdf', 'rvs']:
    method = multivariate_t_gen.__dict__[name]
    method_frozen = multivariate_t_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_params)