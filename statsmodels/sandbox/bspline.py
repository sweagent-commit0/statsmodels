"""
Bspines and smoothing splines.

General references:

    Craven, P. and Wahba, G. (1978) "Smoothing noisy data with spline functions.
    Estimating the correct degree of smoothing by
    the method of generalized cross-validation."
    Numerische Mathematik, 31(4), 377-403.

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.

    Hutchison, M. and Hoog, F. "Smoothing noisy data with spline functions."
    Numerische Mathematik, 47(1), 99-106.
"""
import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline
import warnings
_msg = '\nThe bspline code is technology preview and requires significant work\non the public API and documentation. The API will likely change in the future\n'
warnings.warn(_msg, FutureWarning)

def _band2array(a, lower=0, symmetric=False, hermitian=False):
    """
    Take an upper or lower triangular banded matrix and return a
    numpy array.

    INPUTS:
       a         -- a matrix in upper or lower triangular banded matrix
       lower     -- is the matrix upper or lower triangular?
       symmetric -- if True, return the original result plus its transpose
       hermitian -- if True (and symmetric False), return the original
                    result plus its conjugate transposed
    """
    pass

def _upper2lower(ub):
    """
    Convert upper triangular banded matrix to lower banded form.

    INPUTS:
       ub  -- an upper triangular banded matrix

    OUTPUTS: lb
       lb  -- a lower triangular banded matrix with same entries
              as ub
    """
    pass

def _lower2upper(lb):
    """
    Convert lower triangular banded matrix to upper banded form.

    INPUTS:
       lb  -- a lower triangular banded matrix

    OUTPUTS: ub
       ub  -- an upper triangular banded matrix with same entries
              as lb
    """
    pass

def _triangle2unit(tb, lower=0):
    """
    Take a banded triangular matrix and return its diagonal and the
    unit matrix: the banded triangular matrix with 1's on the diagonal,
    i.e. each row is divided by the corresponding entry on the diagonal.

    INPUTS:
       tb    -- a lower triangular banded matrix
       lower -- if True, then tb is assumed to be lower triangular banded,
                in which case return value is also lower triangular banded.

    OUTPUTS: d, b
       d     -- diagonal entries of tb
       b     -- unit matrix: if lower is False, b is upper triangular
                banded and its rows of have been divided by d,
                else lower is True, b is lower triangular banded
                and its columns have been divieed by d.
    """
    pass

def _trace_symbanded(a, b, lower=0):
    """
    Compute the trace(ab) for two upper or banded real symmetric matrices
    stored either in either upper or lower form.

    INPUTS:
       a, b    -- two banded real symmetric matrices (either lower or upper)
       lower   -- if True, a and b are assumed to be the lower half


    OUTPUTS: trace
       trace   -- trace(ab)
    """
    pass

def _zero_triband(a, lower=0):
    """
    Explicitly zero out unused elements of a real symmetric banded matrix.

    INPUTS:
       a   -- a real symmetric banded matrix (either upper or lower hald)
       lower   -- if True, a is assumed to be the lower half
    """
    pass

class BSpline:
    """

    Bsplines of a given order and specified knots.

    Implementation is based on description in Chapter 5 of

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.


    INPUTS:
       knots  -- a sorted array of knots with knots[0] the lower boundary,
                 knots[1] the upper boundary and knots[1:-1] the internal
                 knots.
       order  -- order of the Bspline, default is 4 which yields cubic
                 splines
       M      -- number of additional boundary knots, if None it defaults
                 to order
       coef   -- an optional array of real-valued coefficients for the Bspline
                 of shape (knots.shape + 2 * (M - 1) - order,).
       x      -- an optional set of x values at which to evaluate the
                 Bspline to avoid extra evaluation in the __call__ method

    """

    def __init__(self, knots, order=4, M=None, coef=None, x=None):
        knots = np.squeeze(np.unique(np.asarray(knots)))
        if knots.ndim != 1:
            raise ValueError('expecting 1d array for knots')
        self.m = order
        if M is None:
            M = self.m
        self.M = M
        self.tau = np.hstack([[knots[0]] * (self.M - 1), knots, [knots[-1]] * (self.M - 1)])
        self.K = knots.shape[0] - 2
        if coef is None:
            self.coef = np.zeros(self.K + 2 * self.M - self.m, np.float64)
        else:
            self.coef = np.squeeze(coef)
            if self.coef.shape != self.K + 2 * self.M - self.m:
                raise ValueError('coefficients of Bspline have incorrect shape')
        if x is not None:
            self.x = x
    x = property(_getx, _setx)

    def __call__(self, *args):
        """
        Evaluate the BSpline at a given point, yielding
        a matrix B and return

        B * self.coef


        INPUTS:
           args -- optional arguments. If None, it returns self._basisx,
                   the BSpline evaluated at the x values passed in __init__.
                   Otherwise, return the BSpline evaluated at the
                   first argument args[0].

        OUTPUTS: y
           y    -- value of Bspline at specified x values

        BUGS:
           If self has no attribute x, an exception will be raised
           because self has no attribute _basisx.
        """
        if not args:
            b = self._basisx.T
        else:
            x = args[0]
            b = np.asarray(self.basis(x)).T
        return np.squeeze(np.dot(b, self.coef))

    def basis_element(self, x, i, d=0):
        """
        Evaluate a particular basis element of the BSpline,
        or its derivative.

        INPUTS:
           x  -- x values at which to evaluate the basis element
           i  -- which element of the BSpline to return
           d  -- the order of derivative

        OUTPUTS: y
           y  -- value of d-th derivative of the i-th basis element
                 of the BSpline at specified x values
        """
        pass

    def basis(self, x, d=0, lower=None, upper=None):
        """
        Evaluate the basis of the BSpline or its derivative.
        If lower or upper is specified, then only
        the [lower:upper] elements of the basis are returned.

        INPUTS:
           x     -- x values at which to evaluate the basis element
           i     -- which element of the BSpline to return
           d     -- the order of derivative
           lower -- optional lower limit of the set of basis
                    elements
           upper -- optional upper limit of the set of basis
                    elements

        OUTPUTS: y
           y  -- value of d-th derivative of the basis elements
                 of the BSpline at specified x values
        """
        pass

    def gram(self, d=0):
        """
        Compute Gram inner product matrix, storing it in lower
        triangular banded form.

        The (i,j) entry is

        G_ij = integral b_i^(d) b_j^(d)

        where b_i are the basis elements of the BSpline and (d) is the
        d-th derivative.

        If d is a matrix then, it is assumed to specify a differential
        operator as follows: the first row represents the order of derivative
        with the second row the coefficient corresponding to that order.

        For instance:

        [[2, 3],
         [3, 1]]

        represents 3 * f^(2) + 1 * f^(3).

        INPUTS:
           d    -- which derivative to apply to each basis element,
                   if d is a matrix, it is assumed to specify
                   a differential operator as above

        OUTPUTS: gram
           gram -- the matrix of inner products of (derivatives)
                   of the BSpline elements
        """
        pass

class SmoothingSpline(BSpline):
    penmax = 30.0
    method = 'target_df'
    target_df = 5
    default_pen = 0.001
    optimize = True
    '\n    A smoothing spline, which can be used to smooth scatterplots, i.e.\n    a list of (x,y) tuples.\n\n    See fit method for more information.\n\n    '

    def fit(self, y, x=None, weights=None, pen=0.0):
        """
        Fit the smoothing spline to a set of (x,y) pairs.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           weights -- optional array of weights
           pen     -- constant in front of Gram matrix

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.

        ALGORITHM:
           Formally, this solves a minimization:

           fhat = ARGMIN_f SUM_i=1^n (y_i-f(x_i))^2 + pen * int f^(2)^2

           int is integral. pen is lambda (from Hastie)

           See Chapter 5 of

           Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
           Learning." Springer-Verlag. 536 pages.

           for more details.

        TODO:
           Should add arbitrary derivative penalty instead of just
           second derivative.
        """
        pass

    def gcv(self):
        """
        Generalized cross-validation score of current fit.

        Craven, P. and Wahba, G.  "Smoothing noisy data with spline functions.
        Estimating the correct degree of smoothing by
        the method of generalized cross-validation."
        Numerische Mathematik, 31(4), 377-403.
        """
        pass

    def df_resid(self):
        """
        Residual degrees of freedom in the fit.

        self.N - self.trace()

        where self.N is the number of observations of last fit.
        """
        pass

    def df_fit(self):
        """
        How many degrees of freedom used in the fit?

        self.trace()
        """
        pass

    def trace(self):
        """
        Trace of the smoothing matrix S(pen)

        TODO: addin a reference to Wahba, and whoever else I used.
        """
        pass

    def fit_target_df(self, y, x=None, df=None, weights=None, tol=0.001, apen=0, bpen=0.001):
        """
        Fit smoothing spline with approximately df degrees of freedom
        used in the fit, i.e. so that self.trace() is approximately df.

        Uses binary search strategy.

        In general, df must be greater than the dimension of the null space
        of the Gram inner product. For cubic smoothing splines, this means
        that df > 2.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           apen    -- lower bound of penalty for binary search
           bpen    -- upper bound of penalty for binary search

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """
        pass

    def fit_optimize_gcv(self, y, x=None, weights=None, tol=0.001, brack=(-100, 20)):
        """
        Fit smoothing spline trying to optimize GCV.

        Try to find a bracketing interval for scipy.optimize.golden
        based on bracket.

        It is probably best to use target_df instead, as it is
        sometimes difficult to find a bracketing interval.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           brack   -- an initial guess at the bracketing interval

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """
        pass