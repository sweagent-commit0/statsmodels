"""
Spline and other smoother classes for Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

Created on Fri Jun  5 16:32:00 2015
"""
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints

def make_bsplines_basis(x, df, degree):
    """ make a spline basis for x """
    pass

def get_knots_bsplines(x=None, df=None, knots=None, degree=3, spacing='quantile', lower_bound=None, upper_bound=None, all_knots=None):
    """knots for use in B-splines

    There are two main options for the knot placement

    - quantile spacing with multiplicity of boundary knots
    - equal spacing extended to boundary or exterior knots

    The first corresponds to splines as used by patsy. the second is the
    knot spacing for P-Splines.
    """
    pass

def _get_integration_points(knots, k_points=3):
    """add points to each subinterval defined by knots

    inserts k_points between each two consecutive knots
    """
    pass

def get_covder2(smoother, k_points=3, integration_points=None, skip_ctransf=False, deriv=2):
    """
    Approximate integral of cross product of second derivative of smoother

    This uses scipy.integrate simps to compute an approximation to the
    integral of the smoother derivative cross-product at knots plus k_points
    in between knots.
    """
    pass

def make_poly_basis(x, degree, intercept=True):
    """
    given a vector x returns poly=(1, x, x^2, ..., x^degree)
    and its first and second derivative
    """
    pass

class UnivariateGamSmoother(with_metaclass(ABCMeta)):
    """Base Class for single smooth component
    """

    def __init__(self, x, constraints=None, variable_name='x'):
        self.x = x
        self.constraints = constraints
        self.variable_name = variable_name
        self.nobs, self.k_variables = (len(x), 1)
        base4 = self._smooth_basis_for_single_variable()
        if constraints == 'center':
            constraints = base4[0].mean(0)[None, :]
        if constraints is not None and (not isinstance(constraints, str)):
            ctransf = transf_constraints(constraints)
            self.ctransf = ctransf
        elif not hasattr(self, 'ctransf'):
            self.ctransf = None
        self.basis, self.der_basis, self.der2_basis, self.cov_der2 = base4
        if self.ctransf is not None:
            ctransf = self.ctransf
            if base4[0] is not None:
                self.basis = base4[0].dot(ctransf)
            if base4[1] is not None:
                self.der_basis = base4[1].dot(ctransf)
            if base4[2] is not None:
                self.der2_basis = base4[2].dot(ctransf)
            if base4[3] is not None:
                self.cov_der2 = ctransf.T.dot(base4[3]).dot(ctransf)
        self.dim_basis = self.basis.shape[1]
        self.col_names = [self.variable_name + '_s' + str(i) for i in range(self.dim_basis)]

class UnivariateGenericSmoother(UnivariateGamSmoother):
    """Generic single smooth component
    """

    def __init__(self, x, basis, der_basis, der2_basis, cov_der2, variable_name='x'):
        self.basis = basis
        self.der_basis = der_basis
        self.der2_basis = der2_basis
        self.cov_der2 = cov_der2
        super(UnivariateGenericSmoother, self).__init__(x, variable_name=variable_name)

class UnivariatePolynomialSmoother(UnivariateGamSmoother):
    """polynomial single smooth component
    """

    def __init__(self, x, degree, variable_name='x'):
        self.degree = degree
        super(UnivariatePolynomialSmoother, self).__init__(x, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        """
        given a vector x returns poly=(1, x, x^2, ..., x^degree)
        and its first and second derivative
        """
        pass

class UnivariateBSplines(UnivariateGamSmoother):
    """B-Spline single smooth component

    This creates and holds the B-Spline basis function for one
    component.

    Parameters
    ----------
    x : ndarray, 1-D
        underlying explanatory variable for smooth terms.
    df : int
        number of basis functions or degrees of freedom
    degree : int
        degree of the spline
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_name : {None, str}
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    covder2_kwds : {None, dict}
        options for computing the penalty matrix from the second derivative
        of the spline.
    knot_kwds : {None, list[dict]}
        option for the knot selection.
        By default knots are selected in the same way as in patsy, however the
        number of knots is independent of keeping or removing the constant.
        Interior knot selection is based on quantiles of the data and is the
        same in patsy and mgcv. Boundary points are at the limits of the data
        range.
        The available options use with `get_knots_bsplines` are

        - knots : None or array
          interior knots
        - spacing : 'quantile' or 'equal'
        - lower_bound : None or float
          location of lower boundary knots, all boundary knots are at the same
          point
        - upper_bound : None or float
          location of upper boundary knots, all boundary knots are at the same
          point
        - all_knots : None or array
          If all knots are provided, then those will be taken as given and
          all other options will be ignored.
    """

    def __init__(self, x, df, degree=3, include_intercept=False, constraints=None, variable_name='x', covder2_kwds=None, **knot_kwds):
        self.degree = degree
        self.df = df
        self.include_intercept = include_intercept
        self.knots = get_knots_bsplines(x, degree=degree, df=df, **knot_kwds)
        self.covder2_kwds = covder2_kwds if covder2_kwds is not None else {}
        super(UnivariateBSplines, self).__init__(x, constraints=constraints, variable_name=variable_name)

    def transform(self, x_new, deriv=0, skip_ctransf=False):
        """create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new : ndarray
            observations of the underlying explanatory variable
        deriv : int
            which derivative of the spline basis to compute
            This is an options for internal computation.
        skip_ctransf : bool
            whether to skip the constraint transform
            This is an options for internal computation.

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``
        """
        pass

class UnivariateCubicSplines(UnivariateGamSmoother):
    """Cubic Spline single smooth component

    Cubic splines as described in the wood's book in chapter 3
    """

    def __init__(self, x, df, constraints=None, transform='domain', variable_name='x'):
        self.degree = 3
        self.df = df
        self.transform_data_method = transform
        self.x = x = self.transform_data(x, initialize=True)
        self.knots = _equally_spaced_knots(x, df)
        super(UnivariateCubicSplines, self).__init__(x, constraints=constraints, variable_name=variable_name)

class UnivariateCubicCyclicSplines(UnivariateGamSmoother):
    """cyclic cubic regression spline single smooth component

    This creates and holds the Cyclic CubicSpline basis function for one
    component.

    Parameters
    ----------
    x : ndarray, 1-D
        underlying explanatory variable for smooth terms.
    df : int
        number of basis functions or degrees of freedom
    degree : int
        degree of the spline
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_name : None or str
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    """

    def __init__(self, x, df, constraints=None, variable_name='x'):
        self.degree = 3
        self.df = df
        self.x = x
        self.knots = _equally_spaced_knots(x, df)
        super(UnivariateCubicCyclicSplines, self).__init__(x, constraints=constraints, variable_name=variable_name)

    def _get_b_and_d(self, knots):
        """Returns mapping of cyclic cubic spline values to 2nd derivatives.

        .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006,
           pp 146-147

        Parameters
        ----------
        knots : ndarray
            The 1-d array knots used for cubic spline parametrization,
            must be sorted in ascending order.

        Returns
        -------
        b : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.
        d : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.

        Notes
        -----
        The penalty matrix is equal to ``s = d.T.dot(b^-1).dot(d)``
        """
        pass

class AdditiveGamSmoother(with_metaclass(ABCMeta)):
    """Base class for additive smooth components
    """

    def __init__(self, x, variable_names=None, include_intercept=False, **kwargs):
        if isinstance(x, pd.DataFrame):
            data_names = x.columns.tolist()
        elif isinstance(x, pd.Series):
            data_names = [x.name]
        else:
            data_names = None
        x = np.asarray(x)
        if x.ndim == 1:
            self.x = x.copy()
            self.x.shape = (len(x), 1)
        else:
            self.x = x
        self.nobs, self.k_variables = self.x.shape
        if isinstance(include_intercept, bool):
            self.include_intercept = [include_intercept] * self.k_variables
        else:
            self.include_intercept = include_intercept
        if variable_names is None:
            if data_names is not None:
                self.variable_names = data_names
            else:
                self.variable_names = ['x' + str(i) for i in range(self.k_variables)]
        else:
            self.variable_names = variable_names
        self.smoothers = self._make_smoothers_list()
        self.basis = np.hstack(list((smoother.basis for smoother in self.smoothers)))
        self.dim_basis = self.basis.shape[1]
        self.penalty_matrices = [smoother.cov_der2 for smoother in self.smoothers]
        self.col_names = []
        for smoother in self.smoothers:
            self.col_names.extend(smoother.col_names)
        self.mask = []
        last_column = 0
        for smoother in self.smoothers:
            mask = np.array([False] * self.dim_basis)
            mask[last_column:smoother.dim_basis + last_column] = True
            last_column = last_column + smoother.dim_basis
            self.mask.append(mask)

    def transform(self, x_new):
        """create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new: ndarray
            observations of the underlying explanatory variable

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``.
        """
        pass

class GenericSmoothers(AdditiveGamSmoother):
    """generic class for additive smooth components for GAM
    """

    def __init__(self, x, smoothers):
        self.smoothers = smoothers
        super(GenericSmoothers, self).__init__(x, variable_names=None)

class PolynomialSmoother(AdditiveGamSmoother):
    """additive polynomial components for GAM
    """

    def __init__(self, x, degrees, variable_names=None):
        self.degrees = degrees
        super(PolynomialSmoother, self).__init__(x, variable_names=variable_names)

class BSplines(AdditiveGamSmoother):
    """additive smooth components using B-Splines

    This creates and holds the B-Spline basis function for several
    components.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  {int, array_like[int]}
        number of basis functions or degrees of freedom; should be equal
        in length to the number of columns of `x`; may be an integer if
        `x` has one column or is 1-D.
    degree : {int, array_like[int]}
        degree(s) of the spline; the same length and type rules apply as
        to `df`
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    knot_kwds : None or list of dict
        option for the knot selection.
        By default knots are selected in the same way as in patsy, however the
        number of knots is independent of keeping or removing the constant.
        Interior knot selection is based on quantiles of the data and is the
        same in patsy and mgcv. Boundary points are at the limits of the data
        range.
        The available options use with `get_knots_bsplines` are

        - knots : None or array
          interior knots
        - spacing : 'quantile' or 'equal'
        - lower_bound : None or float
          location of lower boundary knots, all boundary knots are at the same
          point
        - upper_bound : None or float
          location of upper boundary knots, all boundary knots are at the same
          point
        - all_knots : None or array
          If all knots are provided, then those will be taken as given and
          all other options will be ignored.


    Attributes
    ----------
    smoothers : list of univariate smooth component instances
    basis : design matrix, array of spline bases columns for all components
    penalty_matrices : list of penalty matrices, one for each smooth term
    dim_basis : number of columns in the basis
    k_variables : number of smooth components
    col_names : created names for the basis columns

    There are additional attributes about the specification of the splines
    and some attributes mainly for internal use.

    Notes
    -----
    A constant in the spline basis function can be removed in two different
    ways.
    The first is by dropping one basis column and normalizing the
    remaining columns. This is obtained by the default
    ``include_intercept=False, constraints=None``
    The second option is by using the centering transform which is a linear
    transformation of all basis functions. As a consequence of the
    transformation, the B-spline basis functions do not have locally bounded
    support anymore. This is obtained ``constraints='center'``. In this case
    ``include_intercept`` will be automatically set to True to avoid
    dropping an additional column.
    """

    def __init__(self, x, df, degree, include_intercept=False, constraints=None, variable_names=None, knot_kwds=None):
        if isinstance(degree, int):
            self.degrees = np.array([degree], dtype=int)
        else:
            self.degrees = degree
        if isinstance(df, int):
            self.dfs = np.array([df], dtype=int)
        else:
            self.dfs = df
        self.knot_kwds = knot_kwds
        self.constraints = constraints
        if constraints == 'center':
            include_intercept = True
        super(BSplines, self).__init__(x, include_intercept=include_intercept, variable_names=variable_names)

class CubicSplines(AdditiveGamSmoother):
    """additive smooth components using cubic splines as in Wood 2006.

    Note, these splines do NOT use the same spline basis as
    ``Cubic Regression Splines``.
    """

    def __init__(self, x, df, constraints='center', transform='domain', variable_names=None):
        self.dfs = df
        self.constraints = constraints
        self.transform = transform
        super(CubicSplines, self).__init__(x, constraints=constraints, variable_names=variable_names)

class CyclicCubicSplines(AdditiveGamSmoother):
    """additive smooth components using cyclic cubic regression splines

    This spline basis is the same as in patsy.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  int
        numer of basis functions or degrees of freedom
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    """

    def __init__(self, x, df, constraints=None, variable_names=None):
        self.dfs = df
        self.constraints = constraints
        super(CyclicCubicSplines, self).__init__(x, variable_names=variable_names)