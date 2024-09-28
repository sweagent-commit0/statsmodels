"""
Covariance models and estimators for GEE.

Some details for the covariance calculations can be found in the Stata
docs:

http://www.stata.com/manuals13/xtxtgee.pdf
"""
from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import ConvergenceWarning, NotImplementedWarning, OutputWarning
from statsmodels.tools.validation import bool_like

class CovStruct:
    """
    Base class for correlation and covariance structures.

    An implementation of this class takes the residuals from a
    regression model that has been fit to grouped data, and uses
    them to estimate the within-group dependence structure of the
    random errors in the model.

    The current state of the covariance structure is represented
    through the value of the `dep_params` attribute.

    The default state of a newly-created instance should always be
    the identity correlation matrix.
    """

    def __init__(self, cov_nearest_method='clipped'):
        self.dep_params = None
        self.cov_adjust = []
        self.cov_nearest_method = cov_nearest_method

    def initialize(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        pass

    def update(self, params):
        """
        Update the association parameter values based on the current
        regression coefficients.

        Parameters
        ----------
        params : array_like
            Working values for the regression parameters.
        """
        pass

    def covariance_matrix(self, endog_expval, index):
        """
        Returns the working covariance or correlation matrix for a
        given cluster of data.

        Parameters
        ----------
        endog_expval : array_like
           The expected values of endog for the cluster for which the
           covariance or correlation matrix will be returned
        index : int
           The index of the cluster for which the covariance or
           correlation matrix will be returned

        Returns
        -------
        M : matrix
            The covariance or correlation matrix of endog
        is_cor : bool
            True if M is a correlation matrix, False if M is a
            covariance matrix
        """
        pass

    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        """
        Solves matrix equations of the form `covmat * soln = rhs` and
        returns the values of `soln`, where `covmat` is the covariance
        matrix represented by this class.

        Parameters
        ----------
        expval : array_like
           The expected value of endog for each observed value in the
           group.
        index : int
           The group index.
        stdev : array_like
            The standard deviation of endog for each observation in
            the group.
        rhs : list/tuple of array_like
            A set of right-hand sides; each defines a matrix equation
            to be solved.

        Returns
        -------
        soln : list/tuple of array_like
            The solutions to the matrix equations.

        Notes
        -----
        Returns None if the solver fails.

        Some dependence structures do not use `expval` and/or `index`
        to determine the correlation matrix.  Some families
        (e.g. binomial) do not use the `stdev` parameter when forming
        the covariance matrix.

        If the covariance matrix is singular or not SPD, it is
        projected to the nearest such matrix.  These projection events
        are recorded in the fit_history attribute of the GEE model.

        Systems of linear equations with the covariance matrix as the
        left hand side (LHS) are solved for different right hand sides
        (RHS); the LHS is only factorized once to save time.

        This is a default implementation, it can be reimplemented in
        subclasses to optimize the linear algebra according to the
        structure of the covariance matrix.
        """
        pass

    def summary(self):
        """
        Returns a text summary of the current estimate of the
        dependence structure.
        """
        pass

class Independence(CovStruct):
    """
    An independence working dependence structure.
    """

class Unstructured(CovStruct):
    """
    An unstructured dependence structure.

    To use the unstructured dependence structure, a `time`
    argument must be provided when creating the GEE.  The
    time argument must be of integer dtype, and indicates
    which position in a complete data vector is occupied
    by each observed value.
    """

    def __init__(self, cov_nearest_method='clipped'):
        super(Unstructured, self).__init__(cov_nearest_method)

class Exchangeable(CovStruct):
    """
    An exchangeable working dependence structure.
    """

    def __init__(self):
        super(Exchangeable, self).__init__()
        self.dep_params = 0.0

class Nested(CovStruct):
    """
    A nested working dependence structure.

    A nested working dependence structure captures unique variance
    associated with each level in a hierarchy of partitions of the
    cases.  For each level of the hierarchy, there is a set of iid
    random effects with mean zero, and with variance that is specific
    to the level.  These variance parameters are estimated from the
    data using the method of moments.

    The top level of the hierarchy is always defined by the required
    `groups` argument to GEE.

    The `dep_data` argument used to create the GEE defines the
    remaining levels of the hierarchy.  it should be either an array,
    or if using the formula interface, a string that contains a
    formula.  If an array, it should contain a `n_obs x k` matrix of
    labels, corresponding to the k levels of partitioning that are
    nested under the top-level `groups` of the GEE instance.  These
    subgroups should be nested from left to right, so that two
    observations with the same label for column j of `dep_data` should
    also have the same label for all columns j' < j (this only applies
    to observations in the same top-level cluster given by the
    `groups` argument to GEE).

    If `dep_data` is a formula, it should usually be of the form `0 +
    a + b + ...`, where `a`, `b`, etc. contain labels defining group
    membership.  The `0 + ` should be included to prevent creation of
    an intercept.  The variable values are interpreted as labels for
    group membership, but the variables should not be explicitly coded
    as categorical, i.e. use `0 + a` not `0 + C(a)`.

    Notes
    -----
    The calculations for the nested structure involve all pairs of
    observations within the top level `group` passed to GEE.  Large
    group sizes will result in slow iterations.
    """

    def initialize(self, model):
        """
        Called on the first call to update

        `ilabels` is a list of n_i x n_i matrices containing integer
        labels that correspond to specific correlation parameters.
        Two elements of ilabels[i] with the same label share identical
        variance components.

        `designx` is a matrix, with each row containing dummy
        variables indicating which variance components are associated
        with the corresponding element of QY.
        """
        pass

    def summary(self):
        """
        Returns a summary string describing the state of the
        dependence structure.
        """
        pass

class Stationary(CovStruct):
    """
    A stationary covariance structure.

    The correlation between two observations is an arbitrary function
    of the distance between them.  Distances up to a given maximum
    value are included in the covariance model.

    Parameters
    ----------
    max_lag : float
        The largest distance that is included in the covariance model.
    grid : bool
        If True, the index positions in the data (after dropping missing
        values) are used to define distances, and the `time` variable is
        ignored.
    """

    def __init__(self, max_lag=1, grid=None):
        super(Stationary, self).__init__()
        grid = bool_like(grid, 'grid', optional=True)
        if grid is None:
            warnings.warn('grid=True will become default in a future version', FutureWarning)
        self.max_lag = max_lag
        self.grid = bool(grid)
        self.dep_params = np.zeros(max_lag + 1)

class Autoregressive(CovStruct):
    """
    A first-order autoregressive working dependence structure.

    The dependence is defined in terms of the `time` component of the
    parent GEE class, which defaults to the index position of each
    value within its cluster, based on the order of values in the
    input data set.  Time represents a potentially multidimensional
    index from which distances between pairs of observations can be
    determined.

    The correlation between two observations in the same cluster is
    dep_params^distance, where `dep_params` contains the (scalar)
    autocorrelation parameter to be estimated, and `distance` is the
    distance between the two observations, calculated from their
    corresponding time values.  `time` is stored as an n_obs x k
    matrix, where `k` represents the number of dimensions in the time
    index.

    The autocorrelation parameter is estimated using weighted
    nonlinear least squares, regressing each value within a cluster on
    each preceding value in the same cluster.

    Parameters
    ----------
    dist_func : function from R^k x R^k to R^+, optional
        A function that computes the distance between the two
        observations based on their `time` values.

    References
    ----------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    def __init__(self, dist_func=None, grid=None):
        super(Autoregressive, self).__init__()
        grid = bool_like(grid, 'grid', optional=True)
        if dist_func is None:
            self.dist_func = lambda x, y: np.abs(x - y).sum()
        else:
            self.dist_func = dist_func
        if grid is None:
            warnings.warn('grid=True will become default in a future version', FutureWarning)
        self.grid = bool(grid)
        if not self.grid:
            self.designx = None
        self.dep_params = 0.0

class CategoricalCovStruct(CovStruct):
    """
    Parent class for covariance structure for categorical data models.

    Attributes
    ----------
    nlevel : int
        The number of distinct levels for the outcome variable.
    ibd : list
        A list whose i^th element ibd[i] is an array whose rows
        contain integer pairs (a,b), where endog_li[i][a:b] is the
        subvector of binary indicators derived from the same ordinal
        value.
    """

class GlobalOddsRatio(CategoricalCovStruct):
    """
    Estimate the global odds ratio for a GEE with ordinal or nominal
    data.

    References
    ----------
    PJ Heagerty and S Zeger. "Marginal Regression Models for Clustered
    Ordinal Measurements". Journal of the American Statistical
    Association Vol. 91, Issue 435 (1996).

    Thomas Lumley. Generalized Estimating Equations for Ordinal Data:
    A Note on Working Correlation Structures. Biometrics Vol. 52,
    No. 1 (Mar., 1996), pp. 354-361
    http://www.jstor.org/stable/2533173

    Notes
    -----
    The following data structures are calculated in the class:

    'ibd' is a list whose i^th element ibd[i] is a sequence of integer
    pairs (a,b), where endog_li[i][a:b] is the subvector of binary
    indicators derived from the same ordinal value.

    `cpp` is a dictionary where cpp[group] is a map from cut-point
    pairs (c,c') to the indices of all between-subject pairs derived
    from the given cut points.
    """

    def __init__(self, endog_type):
        super(GlobalOddsRatio, self).__init__()
        self.endog_type = endog_type
        self.dep_params = 0.0

    def pooled_odds_ratio(self, tables):
        """
        Returns the pooled odds ratio for a list of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average
        of the sample odds ratios of the tables.
        """
        pass

    def observed_crude_oddsratio(self):
        """
        To obtain the crude (global) odds ratio, first pool all binary
        indicators corresponding to a given pair of cut points (c,c'),
        then calculate the odds ratio for this 2x2 table.  The crude
        odds ratio is the inverse variance weighted average of these
        odds ratios.  Since the covariate effects are ignored, this OR
        will generally be greater than the stratified OR.
        """
        pass

    def get_eyy(self, endog_expval, index):
        """
        Returns a matrix V such that V[i,j] is the joint probability
        that endog[i] = 1 and endog[j] = 1, based on the marginal
        probabilities of endog and the global odds ratio `current_or`.
        """
        pass

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        """
        Update the global odds ratio based on the current value of
        params.
        """
        pass

class OrdinalIndependence(CategoricalCovStruct):
    """
    An independence covariance structure for ordinal models.

    The working covariance between indicators derived from different
    observations is zero.  The working covariance between indicators
    derived form a common observation is determined from their current
    mean values.

    There are no parameters to estimate in this covariance structure.
    """

class NominalIndependence(CategoricalCovStruct):
    """
    An independence covariance structure for nominal models.

    The working covariance between indicators derived from different
    observations is zero.  The working covariance between indicators
    derived form a common observation is determined from their current
    mean values.

    There are no parameters to estimate in this covariance structure.
    """

class Equivalence(CovStruct):
    """
    A covariance structure defined in terms of equivalence classes.

    An 'equivalence class' is a set of pairs of observations such that
    the covariance of every pair within the equivalence class has a
    common value.

    Parameters
    ----------
    pairs : dict-like
      A dictionary of dictionaries, where `pairs[group][label]`
      provides the indices of all pairs of observations in the group
      that have the same covariance value.  Specifically,
      `pairs[group][label]` is a tuple `(j1, j2)`, where `j1` and `j2`
      are integer arrays of the same length.  `j1[i], j2[i]` is one
      index pair that belongs to the `label` equivalence class.  Only
      one triangle of each covariance matrix should be included.
      Positions where j1 and j2 have the same value are variance
      parameters.
    labels : array_like
      An array of labels such that every distinct pair of labels
      defines an equivalence class.  Either `labels` or `pairs` must
      be provided.  When the two labels in a pair are equal two
      equivalence classes are defined: one for the diagonal elements
      (corresponding to variances) and one for the off-diagonal
      elements (corresponding to covariances).
    return_cov : bool
      If True, `covariance_matrix` returns an estimate of the
      covariance matrix, otherwise returns an estimate of the
      correlation matrix.

    Notes
    -----
    Using `labels` to define the class is much easier than using
    `pairs`, but is less general.

    Any pair of values not contained in `pairs` will be assigned zero
    covariance.

    The index values in `pairs` are row indices into the `exog`
    matrix.  They are not updated if missing data are present.  When
    using this covariance structure, missing data should be removed
    before constructing the model.

    If using `labels`, after a model is defined using the covariance
    structure it is possible to remove a label pair from the second
    level of the `pairs` dictionary to force the corresponding
    covariance to be zero.

    Examples
    --------
    The following sets up the `pairs` dictionary for a model with two
    groups, equal variance for all observations, and constant
    covariance for all pairs of observations within each group.

    >> pairs = {0: {}, 1: {}}
    >> pairs[0][0] = (np.r_[0, 1, 2], np.r_[0, 1, 2])
    >> pairs[0][1] = np.tril_indices(3, -1)
    >> pairs[1][0] = (np.r_[3, 4, 5], np.r_[3, 4, 5])
    >> pairs[1][2] = 3 + np.tril_indices(3, -1)
    """

    def __init__(self, pairs=None, labels=None, return_cov=False):
        super(Equivalence, self).__init__()
        if pairs is None and labels is None:
            raise ValueError('Equivalence cov_struct requires either `pairs` or `labels`')
        if pairs is not None and labels is not None:
            raise ValueError('Equivalence cov_struct accepts only one of `pairs` and `labels`')
        if pairs is not None:
            import copy
            self.pairs = copy.deepcopy(pairs)
        if labels is not None:
            self.labels = np.asarray(labels)
        self.return_cov = return_cov

    def _make_pairs(self, i, j):
        """
        Create arrays containing all unique ordered pairs of i, j.

        The arrays i and j must be one-dimensional containing non-negative
        integers.
        """
        pass