"""
Linear mixed effects models are regression models for dependent data.
They can be used to estimate regression relationships involving both
means and variances.

These models are also known as multilevel linear models, and
hierarchical linear models.

The MixedLM class fits linear mixed effects models to data, and
provides support for some common post-estimation tasks.  This is a
group-based implementation that is most efficient for models in which
the data can be partitioned into independent groups.  Some models with
crossed effects can be handled by specifying a model with a single
group.

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* n_i is the number of observations in group i

* Y is a n_i dimensional response vector (called endog in MixedLM)

* X is a n_i x k_fe dimensional design matrix for the fixed effects
  (called exog in MixedLM)

* beta is a k_fe-dimensional vector of fixed effects parameters
  (called fe_params in MixedLM)

* Z is a design matrix for the random effects with n_i rows (called
  exog_re in MixedLM).  The number of columns in Z can vary by group
  as discussed below.

* gamma is a random vector with mean 0.  The covariance matrix for the
  first `k_re` elements of `gamma` (called cov_re in MixedLM) is
  common to all groups.  The remaining elements of `gamma` are
  variance components as discussed in more detail below. Each group
  receives its own independent realization of gamma.

* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The marginal mean structure is E[Y | X, Z] = X*beta.  If only the mean
structure is of interest, GEE is an alternative to using linear mixed
models.

Two types of random effects are supported.  Standard random effects
are correlated with each other in arbitrary ways.  Every group has the
same number (`k_re`) of standard random effects, with the same joint
distribution (but with independent realizations across the groups).

Variance components are uncorrelated with each other, and with the
standard random effects.  Each variance component has mean zero, and
all realizations of a given variance component have the same variance
parameter.  The number of realized variance components per variance
parameter can differ across the groups.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

See also this more recent document:

http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates 1988, adapted to support variance components.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `scale` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is scale*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects in one group.

* `vcomp` is a vector of variance parameters.  The length of `vcomp`
  is determined by the number of keys in either the `exog_vc` argument
  to ``MixedLM``, or the `vc_formula` argument when using formulas to
  fit a model.

Notes:

1. Three different parameterizations are used in different places.
The regression slopes (usually called `fe_params`) are identical in
all three parameterizations, but the variance parameters differ.  The
parameterizations are:

* The "user parameterization" in which cov(endog) = scale*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "user" cov_re is
  equal to the "profile" cov_re1 times the scale.

* The "square root parameterization" in which we work with the Cholesky
  factor of cov_re1 instead of cov_re directly.  This is hidden from the
  user.

All three parameterizations can be packed into a vector by
(optionally) concatenating `fe_params` together with the lower
triangle or Cholesky square root of the dependence structure, followed
by the variance parameters for the variance components.  The are
stored as square roots if (and only if) the random effects covariance
matrix is stored as its Cholesky factor.  Note that when unpacking, it
is important to either square or reflect the dependence structure
depending on which parameterization is being used.

Two score methods are implemented.  One takes the score with respect
to the elements of the random effects covariance matrix (used for
inference once the MLE is reached), and the other takes the score with
respect to the parameters of the Cholesky square root of the random
effects covariance matrix (used for optimization).

The numerical optimization uses GLS to avoid explicitly optimizing
over the fixed effects parameters.  The likelihood that is optimized
is profiled over both the scale parameter (a scalar) and the fixed
effects parameters (if any).  As a result of this profiling, it is
difficult and unnecessary to calculate the Hessian of the profiled log
likelihood function, so that calculation is not implemented here.
Therefore, optimization methods requiring the Hessian matrix such as
the Newton-Raphson algorithm cannot be used for model fitting.
"""
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
_warn_cov_sing = 'The random effects covariance matrix is singular.'

def _dot(x, y):
    """
    Returns the dot product of the arrays, works for sparse and dense.
    """
    pass

def _multi_dot_three(A, B, C):
    """
    Find best ordering for three arrays and do the multiplication.

    Doing in manually instead of using dynamic programing is
    approximately 15 times faster.
    """
    pass

def _dotsum(x, y):
    """
    Returns sum(x * y), where '*' is the pointwise product, computed
    efficiently for dense and sparse matrices.
    """
    pass

class VCSpec:
    """
    Define the variance component structure of a multilevel model.

    An instance of the class contains three attributes:

    - names : names[k] is the name of variance component k.

    - mats : mats[k][i] is the design matrix for group index
      i in variance component k.

    - colnames : colnames[k][i] is the list of column names for
      mats[k][i].

    The groups in colnames and mats must be in sorted order.
    """

    def __init__(self, names, colnames, mats):
        self.names = names
        self.colnames = colnames
        self.mats = mats

def _get_exog_re_names(self, exog_re):
    """
    Passes through if given a list of names. Otherwise, gets pandas names
    or creates some generic variable names as needed.
    """
    pass

class MixedLMParams:
    """
    This class represents a parameter state for a mixed linear model.

    Parameters
    ----------
    k_fe : int
        The number of covariates with fixed effects.
    k_re : int
        The number of covariates with random coefficients (excluding
        variance components).
    k_vc : int
        The number of variance components parameters.

    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, k_vc):
        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) // 2
        self.k_vc = k_vc
        self.k_tot = self.k_fe + self.k_re2 + self.k_vc
        self._ix = np.tril_indices(self.k_re)

    def from_packed(params, k_fe, k_re, use_sqrt, has_fe):
        """
        Create a MixedLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array_like
            The mode parameters packed into a single vector.
        k_fe : int
            The number of covariates with fixed effects
        k_re : int
            The number of covariates with random effects (excluding
            variance components).
        use_sqrt : bool
            If True, the random effects covariance matrix is provided
            as its Cholesky factor, otherwise the lower triangle of
            the covariance matrix is stored.
        has_fe : bool
            If True, `params` contains fixed effects parameters.
            Otherwise, the fixed effects parameters are set to zero.

        Returns
        -------
        A MixedLMParams object.
        """
        pass
    from_packed = staticmethod(from_packed)

    def from_components(fe_params=None, cov_re=None, cov_re_sqrt=None, vcomp=None):
        """
        Create a MixedLMParams object from each parameter component.

        Parameters
        ----------
        fe_params : array_like
            The fixed effects parameter (a 1-dimensional array).  If
            None, there are no fixed effects.
        cov_re : array_like
            The random effects covariance matrix (a square, symmetric
            2-dimensional array).
        cov_re_sqrt : array_like
            The Cholesky (lower triangular) square root of the random
            effects covariance matrix.
        vcomp : array_like
            The variance component parameters.  If None, there are no
            variance components.

        Returns
        -------
        A MixedLMParams object.
        """
        pass
    from_components = staticmethod(from_components)

    def copy(self):
        """
        Returns a copy of the object.
        """
        pass

    def get_packed(self, use_sqrt, has_fe=False):
        """
        Return the model parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : bool
            If True, the Cholesky square root of `cov_re` is
            included in the packed result.  Otherwise the
            lower triangle of `cov_re` is included.
        has_fe : bool
            If True, the fixed effects parameters are included
            in the packed result, otherwise they are omitted.
        """
        pass

def _smw_solver(s, A, AtA, Qi, di):
    """
    Returns a solver for the linear system:

    .. math::

        (sI + ABA^\\prime) y = x

    The returned function f satisfies f(x) = y as defined above.

    B and its inverse matrix are block diagonal.  The upper left block
    of :math:`B^{-1}` is Qi and its lower right block is diag(di).

    Parameters
    ----------
    s : scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p, may be sparse.
    AtA : square ndarray
        :math:`A^\\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.

    Returns
    -------
    A function for solving a linear system, as documented above.

    Notes
    -----
    Uses Sherman-Morrison-Woodbury identity:
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """
    pass

def _smw_logdet(s, A, AtA, Qi, di, B_logdet):
    """
    Returns the log determinant of

    .. math::

        sI + ABA^\\prime

    Uses the matrix determinant lemma to accelerate the calculation.
    B is assumed to be positive definite, and s > 0, therefore the
    determinant is positive.

    Parameters
    ----------
    s : positive scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p.
    AtA : square ndarray
        :math:`A^\\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.
    B_logdet : real
        The log determinant of B

    Returns
    -------
    The log determinant of s*I + A*B*A'.

    Notes
    -----
    Uses the matrix determinant lemma:
        https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """
    pass

class MixedLM(base.LikelihoodModel):
    """
    Linear Mixed Effects Model

    Parameters
    ----------
    endog : 1d array_like
        The dependent variable
    exog : 2d array_like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array_like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array_like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : VCSpec instance or dict-like (deprecated)
        A VCSPec instance defines the structure of the variance
        components in the model.  Alternatively, see notes below
        for a dictionary-based format.  The dictionary format is
        deprecated and may be removed at some point in the future.
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : str
        The approach to missing data handling

    Notes
    -----
    If `exog_vc` is not a `VCSpec` instance, then it must be a
    dictionary of dictionaries.  Specifically, `exog_vc[a][g]` is a
    matrix whose columns are linearly combined using independent
    random coefficients.  This random term then contributes to the
    variance structure of the data for group `g`.  The random
    coefficients all have mean zero, and have the same variance.  The
    matrix must be `m x k`, where `m` is the number of observations in
    group `g`.  The number of columns may differ among the top-level
    groups.

    The covariates in `exog`, `exog_re` and `exog_vc` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.

    Examples
    --------
    A basic mixed model with fixed effects for the columns of
    ``exog`` and a random intercept for each distinct value of
    ``group``:

    >>> model = sm.MixedLM(endog, exog, groups)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    correlated random coefficients for the columns of ``exog_re``:

    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    independent random coefficients for the columns of ``exog_re``:

    >>> free = MixedLMParams.from_components(
                     fe_params=np.ones(exog.shape[1]),
                     cov_re=np.eye(exog_re.shape[1]))
    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit(free=free)

    A different way to specify independent random coefficients for the
    columns of ``exog_re``.  In this example ``groups`` must be a
    Pandas Series with compatible indexing with ``exog_re``, and
    ``exog_re`` has two columns.

    >>> g = pd.groupby(groups, by=groups).groups
    >>> vc = {}
    >>> vc['1'] = {k : exog_re.loc[g[k], 0] for k in g}
    >>> vc['2'] = {k : exog_re.loc[g[k], 1] for k in g}
    >>> model = sm.MixedLM(endog, exog, groups, vcomp=vc)
    >>> result = model.fit()
    """

    def __init__(self, endog, exog, groups, exog_re=None, exog_vc=None, use_sqrt=True, missing='none', **kwargs):
        _allowed_kwargs = ['missing_idx', 'design_info', 'formula']
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError('argument %s not permitted for MixedLM initialization' % x)
        self.use_sqrt = use_sqrt
        self.reml = True
        self.fe_pen = None
        self.re_pen = None
        if isinstance(exog_vc, dict):
            warnings.warn('Using deprecated variance components format')
            exog_vc = _convert_vc(exog_vc)
        if exog_vc is not None:
            self.k_vc = len(exog_vc.names)
            self.exog_vc = exog_vc
        else:
            self.k_vc = 0
            self.exog_vc = VCSpec([], [], [])
        if exog is not None and data_tools._is_using_ndarray_type(exog, None) and (exog.ndim == 1):
            exog = exog[:, None]
        if exog_re is not None and data_tools._is_using_ndarray_type(exog_re, None) and (exog_re.ndim == 1):
            exog_re = exog_re[:, None]
        super(MixedLM, self).__init__(endog, exog, groups=groups, exog_re=exog_re, missing=missing, **kwargs)
        self._init_keys.extend(['use_sqrt', 'exog_vc'])
        self.k_fe = exog.shape[1]
        if exog_re is None and len(self.exog_vc.names) == 0:
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            names = ['Group Var']
            self.data.param_names = self.exog_names + names
            self.data.exog_re_names = names
            self.data.exog_re_names_full = names
        elif exog_re is not None:
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            self.k_re = self.exog_re.shape[1]
            self.k_re2 = self.k_re * (self.k_re + 1) // 2
        else:
            self.k_re = 0
            self.k_re2 = 0
        if not self.data._param_names:
            param_names, exog_re_names, exog_re_names_full = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full
        self.k_params = self.k_fe + self.k_re2
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict(((s, []) for s in group_labels))
        for i, g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]
        self.nobs = len(self.endog)
        self.n_totobs = self.nobs
        if self.exog_names is None:
            self.exog_names = ['FE%d' % (k + 1) for k in range(self.exog.shape[1])]
        self._aex_r = []
        self._aex_r2 = []
        for i in range(self.n_groups):
            a = self._augment_exog(i)
            self._aex_r.append(a)
            ma = _dot(a.T, a)
            self._aex_r2.append(ma)
        self._lin, self._quad = self._reparam()

    def _make_param_names(self, exog_re):
        """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
        pass

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None, subset=None, use_sparse=False, missing='none', *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array_like
            The data for the model. See Notes.
        re_formula : str
            A one-sided formula defining the variance structure of the
            model.  The default gives a random intercept for each
            group.
        vc_formula : dict-like
            Formulas describing variance components.  `vc_formula[vc]` is
            the formula for the component with variance parameter named
            `vc`.  The formula is processed into a matrix, and the columns
            of this matrix are linearly combined with independent random
            coefficients having mean zero and a common variance.
        subset : array_like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        missing : str
            Either 'none' or 'drop'
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        -----
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        If the variance component is intended to produce random
        intercepts for disjoint subsets of a group, specified by
        string labels or a categorical data value, always use '0 +' in
        the formula so that no overall intercept is included.

        If the variance components specify random slopes and you do
        not also want a random group-level intercept in the model,
        then use '0 +' in the formula to exclude the intercept.

        The variance components formulas are processed separately for
        each group.  If a variable is categorical the results will not
        be affected by whether the group labels are distinct or
        re-used over the top-level groups.

        Examples
        --------
        Suppose we have data from an educational study with students
        nested in classrooms nested in schools.  The students take a
        test, and we want to relate the test scores to the students'
        ages, while accounting for the effects of classrooms and
        schools.  The school will be the top-level group, and the
        classroom is a nested group that is specified as a variance
        component.  Note that the schools may have different number of
        classrooms, and the classroom labels may (but need not be)
        different across the schools.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age', vc_formula=vc,                                   re_formula='1', groups='school', data=data)

        Now suppose we also have a previous test score called
        'pretest'.  If we want the relationship between pretest
        scores and the current test to vary by classroom, we can
        specify a random slope for the pretest score

        >>> vc = {'classroom': '0 + C(classroom)', 'pretest': '0 + pretest'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc,                                   re_formula='1', groups='school', data=data)

        The following model is almost equivalent to the previous one,
        but here the classroom random intercept and pretest slope may
        be correlated.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc,                                   re_formula='1 + pretest', groups='school',                                   data=data)
        """
        pass

    def predict(self, params, exog=None):
        """
        Return predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a mixed linear model.  Can be either a
            MixedLMParams instance, or a vector containing the packed
            model parameters in which the fixed effects parameters are
            at the beginning of the vector, or a vector containing
            only the fixed effects parameters.
        exog : array_like, optional
            Design / exogenous data for the fixed effects. Model exog
            is used if None.

        Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        pass

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """
        pass

    def fit_regularized(self, start_params=None, method='l1', alpha=0, ceps=0.0001, ptol=1e-06, maxit=200, **fit_kwargs):
        """
        Fit a model in which the fixed effects parameters are
        penalized.  The dependence parameters are held fixed at their
        estimated values in the unpenalized model.

        Parameters
        ----------
        method : str of Penalty object
            Method for regularization.  If a string, must be 'l1'.
        alpha : array_like
            Scalar or vector of penalty weights.  If a scalar, the
            same weight is applied to all coefficients; if a vector,
            it contains a weight for each coefficient.  If method is a
            Penalty object, the weights are scaled by alpha.  For L1
            regularization, the weights are used directly.
        ceps : positive real scalar
            Fixed effects parameters smaller than this value
            in magnitude are treated as being zero.
        ptol : positive real scalar
            Convergence occurs when the sup norm difference
            between successive values of `fe_params` is less than
            `ptol`.
        maxit : int
            The maximum number of iterations.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance containing the results.

        Notes
        -----
        The covariance structure is not updated as the fixed effects
        parameters are varied.

        The algorithm used here for L1 regularization is a"shooting"
        or cyclic coordinate descent algorithm.

        If method is 'l1', then `fe_pen` and `cov_pen` are used to
        obtain the covariance structure, but are ignored during the
        L1-penalized fitting.

        References
        ----------
        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized
        Paths for Generalized Linear Models via Coordinate
        Descent. Journal of Statistical Software, 33(1) (2008)
        http://www.jstatsoft.org/v33/i01/paper

        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf
        """
        pass

    def get_fe_params(self, cov_re, vcomp, tol=1e-10):
        """
        Use GLS to update the fixed effects parameter estimates.

        Parameters
        ----------
        cov_re : array_like (2d)
            The covariance matrix of the random effects.
        vcomp : array_like (1d)
            The variance components.
        tol : float
            A tolerance parameter to determine when covariances
            are singular.

        Returns
        -------
        params : ndarray
            The GLS estimates of the fixed effects parameters.
        singular : bool
            True if the covariance is singular
        """
        pass

    def _reparam(self):
        """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variance components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """
        pass

    def _expand_vcomp(self, vcomp, group_ix):
        """
        Replicate variance parameters to match a group's design.

        Parameters
        ----------
        vcomp : array_like
            The variance parameters for the variance components.
        group_ix : int
            The group index

        Returns an expanded version of vcomp, in which each variance
        parameter is copied as many times as there are independent
        realizations of the variance component in the given group.
        """
        pass

    def _augment_exog(self, group_ix):
        """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix for a given group.
        """
        pass

    def loglike(self, params, profile_fe=True):
        """
        Evaluate the (profile) log-likelihood of the linear mixed
        effects model.

        Parameters
        ----------
        params : MixedLMParams, or array_like.
            The parameter value.  If array-like, must be a packed
            parameter vector containing only the covariance
            parameters.
        profile_fe : bool
            If True, replace the provided value of `fe_params` with
            the GLS estimates.

        Returns
        -------
        The log-likelihood value at `params`.

        Notes
        -----
        The scale parameter `scale` is always profiled out of the
        log-likelihood.  In addition, if `profile_fe` is true the
        fixed effects parameters are also profiled out.
        """
        pass

    def _gen_dV_dPar(self, ex_r, solver, group_ix, max_ix=None):
        """
        A generator that yields the element-wise derivative of the
        marginal covariance matrix with respect to the random effects
        variance and covariance parameters.

        ex_r : array_like
            The random effects design matrix
        solver : function
            A function that given x returns V^{-1}x, where V
            is the group's marginal covariance matrix.
        group_ix : int
            The group index
        max_ix : {int, None}
            If not None, the generator ends when this index
            is reached.
        """
        pass

    def score(self, params, profile_fe=True):
        """
        Returns the score vector of the profile log-likelihood.

        Notes
        -----
        The score vector that is returned is computed with respect to
        the parameterization defined by this model instance's
        `use_sqrt` attribute.
        """
        pass

    def score_full(self, params, calc_fe):
        """
        Returns the score with respect to untransformed parameters.

        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed random effects
            parameters (cov_re and vcomp) without fe_params.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.

        Notes
        -----
        `score_re` is taken with respect to the parameterization in
        which `cov_re` is represented through its lower triangle
        (without taking the Cholesky square root).
        """
        pass

    def score_sqrt(self, params, calc_fe=True):
        """
        Returns the score with respect to transformed parameters.

        Calculates the score vector with respect to the
        parameterization in which the random effects covariance matrix
        is represented through its Cholesky square root.

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters.  If array-like must contain packed
            parameters that are compatible with this model instance.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.
        """
        pass

    def hessian(self, params):
        """
        Returns the model's Hessian matrix.

        Calculates the Hessian matrix for the linear mixed effects
        model with respect to the parameterization in which the
        covariance matrix is represented directly (without square-root
        transformation).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters at which the Hessian is calculated.
            If array-like, must contain the packed parameters in a
            form that is compatible with this model instance.

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        sing : boolean
            If True, the covariance matrix is singular and a
            pseudo-inverse is returned.
        """
        pass

    def get_scale(self, fe_params, cov_re, vcomp):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Parameters
        ----------
        fe_params : array_like
            The regression slope estimates
        cov_re : 2d array_like
            Estimate of the random effects covariance matrix
        vcomp : array_like
            Estimate of the variance components

        Returns
        -------
        scale : float
            The estimated error variance.
        """
        pass

    def fit(self, start_params=None, reml=True, niter_sa=0, do_cg=True, fe_pen=None, cov_pen=None, free=None, full_output=False, method=None, **fit_kwargs):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start_params : array_like or MixedLMParams
            Starting values for the profile log-likelihood.  If not a
            `MixedLMParams` instance, this should be an array
            containing the packed parameters for the profile
            log-likelihood, including the fixed effects
            parameters.
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        niter_sa : int
            Currently this argument is ignored and has no effect
            on the results.
        cov_pen : CovariancePenalty object
            A penalty for the random effects covariance matrix
        do_cg : bool, defaults to True
            If False, the optimization is skipped and a results
            object at the given (or default) starting values is
            returned.
        fe_pen : Penalty object
            A penalty on the fixed effects
        free : MixedLMParams object
            If not `None`, this is a mask that allows parameters to be
            held fixed at specified values.  A 1 indicates that the
            corresponding parameter is estimated, a 0 indicates that
            it is fixed at its starting value.  Setting the `cov_re`
            component to the identity matrix fits a model with
            independent random effects.  Note that some optimization
            methods do not respect this constraint (bfgs and lbfgs both
            work).
        full_output : bool
            If true, attach iteration history to results
        method : str
            Optimization method.  Can be a scipy.optimize method name,
            or a list of such names to be tried in sequence.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance.
        """
        pass

class _mixedlm_distribution:
    """
    A private class for simulating data from a given mixed linear model.

    Parameters
    ----------
    model : MixedLM instance
        A mixed linear model
    params : array_like
        A parameter vector defining a mixed linear model.  See
        notes for more information.
    scale : scalar
        The unexplained variance
    exog : array_like
        An array of fixed effect covariates.  If None, model.exog
        is used.

    Notes
    -----
    The params array is a vector containing fixed effects parameters,
    random effects parameters, and variance component parameters, in
    that order.  The lower triangle of the random effects covariance
    matrix is stored.  The random effects and variance components
    parameters are divided by the scale parameter.

    This class is used in Mediation, and possibly elsewhere.
    """

    def __init__(self, model, params, scale, exog):
        self.model = model
        self.exog = exog if exog is not None else model.exog
        po = MixedLMParams.from_packed(params, model.k_fe, model.k_re, False, True)
        self.fe_params = po.fe_params
        self.cov_re = scale * po.cov_re
        self.vcomp = scale * po.vcomp
        self.scale = scale
        group_idx = np.zeros(model.nobs, dtype=int)
        for k, g in enumerate(model.group_labels):
            group_idx[model.row_indices[g]] = k
        self.group_idx = group_idx

    def rvs(self, n):
        """
        Return a vector of simulated values from a mixed linear
        model.

        The parameter n is ignored, but required by the interface
        """
        pass

class MixedLMResults(base.LikelihoodModelResults, base.ResultMixin):
    """
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        Pointer to MixedLM model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        A packed parameter vector for the profile parameterization.
        The first `k_fe` elements are the estimated fixed effects
        coefficients.  The remaining elements are the estimated
        variance parameters.  The variance parameters are all divided
        by `scale` and are not the variance parameters shown
        in the summary.
    fe_params : ndarray
        The fitted fixed-effects coefficients
    cov_re : ndarray
        The fitted random-effects covariance matrix
    bse_fe : ndarray
        The standard errors of the fitted fixed effects coefficients
    bse_re : ndarray
        The standard errors of the fitted random effects covariance
        matrix and variance components.  The first `k_re * (k_re + 1)`
        parameters are the standard errors for the lower triangle of
        `cov_re`, the remaining elements are the standard errors for
        the variance components.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params):
        super(MixedLMResults, self).__init__(model, params, normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np.linalg.matrix_rank(self.model.exog)

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values for the model.

        The fitted values reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        pass

    @cache_readonly
    def resid(self):
        """
        Returns the residuals for the model.

        The residuals reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        pass

    @cache_readonly
    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        pass

    @cache_readonly
    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.

        The first `k_re x (k_re + 1)` elements of the returned array
        are the standard errors of the lower triangle of `cov_re`.
        The remaining elements are the standard errors of the variance
        components.

        Note that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        or p-values if used in the usual way.
        """
        pass

    @cache_readonly
    def random_effects(self):
        """
        The conditional means of random effects given the data.

        Returns
        -------
        random_effects : dict
            A dictionary mapping the distinct `group` values to the
            conditional means of the random effects for the group
            given the data.
        """
        pass

    @cache_readonly
    def random_effects_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """
        pass

    def t_test(self, r_matrix, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array_like
            If an array is given, a p x k 2d array or length k 1d
            array specifying the linear restrictions. It is assumed
            that the linear combination is equal to zero.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.
        """
        pass

    def summary(self, yname=None, xname_fe=None, xname_re=None, title=None, alpha=0.05):
        """
        Summarize the mixed model regression results.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname_fe : list[str], optional
            Fixed effects covariate names
        xname_re : list[str], optional
            Random effects covariate names
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        pass

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        pass

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        pass

    def profile_re(self, re_ix, vtype, num_low=5, dist_low=1.0, num_high=5, dist_high=1.0, **fit_kwargs):
        """
        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : int
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : str
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : int
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : int
            The number of points at which to calculate the likelihood
            above the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        """
        pass

class MixedLMResultsWrapper(base.LikelihoodResultsWrapper):
    _attrs = {'bse_re': ('generic_columns', 'exog_re_names_full'), 'fe_params': ('generic_columns', 'xnames'), 'bse_fe': ('generic_columns', 'xnames'), 'cov_re': ('generic_columns_2d', 'exog_re_names'), 'cov_re_unscaled': ('generic_columns_2d', 'exog_re_names')}
    _upstream_attrs = base.LikelihoodResultsWrapper._wrap_attrs
    _wrap_attrs = base.wrap.union_dicts(_attrs, _upstream_attrs)
    _methods = {}
    _upstream_methods = base.LikelihoodResultsWrapper._wrap_methods
    _wrap_methods = base.wrap.union_dicts(_methods, _upstream_methods)