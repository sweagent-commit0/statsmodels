"""
Bayesian inference for generalized linear mixed models.

Currently only families without additional scale or shape parameters
are supported (binomial and Poisson).

Two estimation approaches are supported: Laplace approximation
('maximum a posteriori'), and variational Bayes (mean field
approximation to the posterior distribution).

All realizations of random effects are modeled to be mutually
independent in this implementation.

The `exog_vc` matrix is the design matrix for the random effects.
Every column of `exog_vc` corresponds to an independent realization of
a random effect.  These random effects have mean zero and an unknown
standard deviation.  The standard deviation parameters are constrained
to be equal within subsets of the columns. When not using formulas,
these subsets are specified through the parameter `ident`.  `ident`
must have the same length as the number of columns of `exog_vc`, and
two columns whose `ident` values are equal have the same standard
deviation.  When formulas are used, the columns of `exog_vc` derived
from a common formula are constrained to have the same standard
deviation.

In many applications, `exog_vc` will be sparse.  A sparse matrix may
be passed when constructing a model class.  If a dense matrix is
passed, it will be converted internally to a sparse matrix.  There
currently is no way to avoid creating a temporary dense version of
`exog_vc` when using formulas.

Model and parameterization
--------------------------
The joint density of data and parameters factors as:

.. math::

    p(y | vc, fep) p(vc | vcp) p(vcp) p(fe)

The terms :math:`p(vcp)` and :math:`p(fe)` are prior distributions
that are taken to be Gaussian (the :math:`vcp` parameters are log
standard deviations so the standard deviations have log-normal
distributions).  The random effects distribution :math:`p(vc | vcp)`
is independent Gaussian (random effect realizations are independent
within and between values of the `ident` array).  The model
:math:`p(y | vc, fep)` depends on the specific GLM being fit.
"""
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
glw = [[0.2955242247147529, -0.1488743389816312], [0.2955242247147529, 0.1488743389816312], [0.2692667193099963, -0.4333953941292472], [0.2692667193099963, 0.4333953941292472], [0.219086362515982, -0.6794095682990244], [0.219086362515982, 0.6794095682990244], [0.1494513491505806, -0.8650633666889845], [0.1494513491505806, 0.8650633666889845], [0.0666713443086881, -0.9739065285171717], [0.0666713443086881, 0.9739065285171717]]
_init_doc = '\n    Generalized Linear Mixed Model with Bayesian estimation\n\n    The class implements the Laplace approximation to the posterior\n    distribution (`fit_map`) and a variational Bayes approximation to\n    the posterior (`fit_vb`).  See the two fit method docstrings for\n    more information about the fitting approaches.\n\n    Parameters\n    ----------\n    endog : array_like\n        Vector of response values.\n    exog : array_like\n        Array of covariates for the fixed effects part of the mean\n        structure.\n    exog_vc : array_like\n        Array of covariates for the random part of the model.  A\n        scipy.sparse array may be provided, or else the passed\n        array will be converted to sparse internally.\n    ident : array_like\n        Array of integer labels showing which random terms (columns\n        of `exog_vc`) have a common variance.\n    vcp_p : float\n        Prior standard deviation for variance component parameters\n        (the prior standard deviation of log(s) is vcp_p, where s is\n        the standard deviation of a random effect).\n    fe_p : float\n        Prior standard deviation for fixed effects parameters.\n    family : statsmodels.genmod.families instance\n        The GLM family.\n    fep_names : list[str]\n        The names of the fixed effects parameters (corresponding to\n        columns of exog).  If None, default names are constructed.\n    vcp_names : list[str]\n        The names of the variance component parameters (corresponding\n        to distinct labels in ident).  If None, default names are\n        constructed.\n    vc_names : list[str]\n        The names of the random effect realizations.\n\n    Returns\n    -------\n    MixedGLMResults object\n\n    Notes\n    -----\n    There are three types of values in the posterior distribution:\n    fixed effects parameters (fep), corresponding to the columns of\n    `exog`, random effects realizations (vc), corresponding to the\n    columns of `exog_vc`, and the standard deviations of the random\n    effects realizations (vcp), corresponding to the unique integer\n    labels in `ident`.\n\n    All random effects are modeled as being independent Gaussian\n    values (given the variance structure parameters).  Every column of\n    `exog_vc` has a distinct realized random effect that is used to\n    form the linear predictors.  The elements of `ident` determine the\n    distinct variance structure parameters.  Two random effect\n    realizations that have the same value in `ident` have the same\n    variance.  When fitting with a formula, `ident` is constructed\n    internally (each element of `vc_formulas` yields a distinct label\n    in `ident`).\n\n    The random effect standard deviation parameters (`vcp`) have\n    log-normal prior distributions with mean 0 and standard deviation\n    `vcp_p`.\n\n    Note that for some families, e.g. Binomial, the posterior mode may\n    be difficult to find numerically if `vcp_p` is set to too large of\n    a value.  Setting `vcp_p` to 0.5 seems to work well.\n\n    The prior for the fixed effects parameters is Gaussian with mean 0\n    and standard deviation `fe_p`.  It is recommended that quantitative\n    covariates be standardized.\n\n    Examples\n    --------{example}\n\n\n    References\n    ----------\n    Introduction to generalized linear mixed models:\n    https://stats.idre.ucla.edu/other/mult-pkg/introduction-to-generalized-linear-mixed-models\n\n    SAS documentation:\n    https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_intromix_a0000000215.htm\n\n    An assessment of estimation methods for generalized linear mixed\n    models with binary outcomes\n    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3866838/\n    '
_logit_example = '\n    A binomial (logistic) random effects model with random intercepts\n    for villages and random slopes for each year within each village:\n\n    >>> random = {"a": \'0 + C(Village)\', "b": \'0 + C(Village)*year_cen\'}\n    >>> model = BinomialBayesMixedGLM.from_formula(\n                   \'y ~ year_cen\', random, data)\n    >>> result = model.fit_vb()\n'
_poisson_example = '\n    A Poisson random effects model with random intercepts for villages\n    and random slopes for each year within each village:\n\n    >>> random = {"a": \'0 + C(Village)\', "b": \'0 + C(Village)*year_cen\'}\n    >>> model = PoissonBayesMixedGLM.from_formula(\n                    \'y ~ year_cen\', random, data)\n    >>> result = model.fit_vb()\n'

class _BayesMixedGLM(base.Model):

    def __init__(self, endog, exog, exog_vc=None, ident=None, family=None, vcp_p=1, fe_p=2, fep_names=None, vcp_names=None, vc_names=None, **kwargs):
        if exog.ndim == 1:
            if isinstance(exog, np.ndarray):
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)
        if exog.ndim != 2:
            msg = "'exog' must have one or two columns"
            raise ValueError(msg)
        if exog_vc.ndim == 1:
            if isinstance(exog_vc, np.ndarray):
                exog_vc = exog_vc[:, None]
            else:
                exog_vc = pd.DataFrame(exog_vc)
        if exog_vc.ndim != 2:
            msg = "'exog_vc' must have one or two columns"
            raise ValueError(msg)
        ident = np.asarray(ident)
        if ident.ndim != 1:
            msg = 'ident must be a one-dimensional array'
            raise ValueError(msg)
        if len(ident) != exog_vc.shape[1]:
            msg = 'len(ident) should match the number of columns of exog_vc'
            raise ValueError(msg)
        if not np.issubdtype(ident.dtype, np.integer):
            msg = 'ident must have an integer dtype'
            raise ValueError(msg)
        if fep_names is None:
            if hasattr(exog, 'columns'):
                fep_names = exog.columns.tolist()
            else:
                fep_names = ['FE_%d' % (k + 1) for k in range(exog.shape[1])]
        if vcp_names is None:
            vcp_names = ['VC_%d' % (k + 1) for k in range(int(max(ident)) + 1)]
        elif len(vcp_names) != len(set(ident)):
            msg = 'The lengths of vcp_names and ident should be the same'
            raise ValueError(msg)
        if not sparse.issparse(exog_vc):
            exog_vc = sparse.csr_matrix(exog_vc)
        ident = ident.astype(int)
        vcp_p = float(vcp_p)
        fe_p = float(fe_p)
        if exog is None:
            k_fep = 0
        else:
            k_fep = exog.shape[1]
        if exog_vc is None:
            k_vc = 0
            k_vcp = 0
        else:
            k_vc = exog_vc.shape[1]
            k_vcp = max(ident) + 1
        exog_vc2 = exog_vc.multiply(exog_vc)
        super(_BayesMixedGLM, self).__init__(endog, exog, **kwargs)
        self.exog_vc = exog_vc
        self.exog_vc2 = exog_vc2
        self.ident = ident
        self.family = family
        self.k_fep = k_fep
        self.k_vc = k_vc
        self.k_vcp = k_vcp
        self.fep_names = fep_names
        self.vcp_names = vcp_names
        self.vc_names = vc_names
        self.fe_p = fe_p
        self.vcp_p = vcp_p
        self.names = fep_names + vcp_names
        if vc_names is not None:
            self.names += vc_names

    def logposterior(self, params):
        """
        The overall log-density: log p(y, fe, vc, vcp).

        This differs by an additive constant from the log posterior
        log p(fe, vc, vcp | y).
        """
        pass

    def logposterior_grad(self, params):
        """
        The gradient of the log posterior.
        """
        pass

    @classmethod
    def from_formula(cls, formula, vc_formulas, data, family=None, vcp_p=1, fe_p=2):
        """
        Fit a BayesMixedGLM using a formula.

        Parameters
        ----------
        formula : str
            Formula for the endog and fixed effects terms (use ~ to
            separate dependent and independent expressions).
        vc_formulas : dictionary
            vc_formulas[name] is a one-sided formula that creates one
            collection of random effects with a common variance
            parameter.  If using categorical (factor) variables to
            produce variance components, note that generally `0 + ...`
            should be used so that an intercept is not included.
        data : data frame
            The data to which the formulas are applied.
        family : genmod.families instance
            A GLM family.
        vcp_p : float
            The prior standard deviation for the logarithms of the standard
            deviations of the random effects.
        fe_p : float
            The prior standard deviation for the fixed effects parameters.
        """
        pass

    def fit(self, method='BFGS', minim_opts=None):
        """
        fit is equivalent to fit_map.

        See fit_map for parameter information.

        Use `fit_vb` to fit the model using variational Bayes.
        """
        pass

    def fit_map(self, method='BFGS', minim_opts=None, scale_fe=False):
        """
        Construct the Laplace approximation to the posterior distribution.

        Parameters
        ----------
        method : str
            Optimization method for finding the posterior mode.
        minim_opts : dict
            Options passed to scipy.minimize.
        scale_fe : bool
            If True, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.

        Returns
        -------
        BayesMixedGLMResults instance.
        """
        pass

    def predict(self, params, exog=None, linear=False):
        """
        Return the fitted mean structure.

        Parameters
        ----------
        params : array_like
            The parameter vector, may be the full parameter vector, or may
            be truncated to include only the mean parameters.
        exog : array_like
            The design matrix for the mean structure.  If omitted, use the
            model's design matrix.
        linear : bool
            If True, return the linear predictor without passing through the
            link function.

        Returns
        -------
        A 1-dimensional array of predicted values
        """
        pass

class _VariationalBayesMixedGLM:
    """
    A mixin providing generic (not family-specific) methods for
    variational Bayes mean field fitting.
    """
    rng = 5
    verbose = False

    def vb_elbo_base(self, h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.

        This function calculates the family-specific ELBO function
        based on information provided from a subclass.

        Parameters
        ----------
        h : function mapping 1d vector to 1d vector
            The contribution of the model to the ELBO function can be
            expressed as y_i*lp_i + Eh_i(z), where y_i and lp_i are
            the response and linear predictor for observation i, and z
            is a standard normal random variable.  This formulation
            can be achieved for any GLM with a canonical link
            function.
        """
        pass

    def vb_elbo_grad_base(self, h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
        """
        Return the gradient of the ELBO function.

        See vb_elbo_base for parameters.
        """
        pass

    def fit_vb(self, mean=None, sd=None, fit_method='BFGS', minim_opts=None, scale_fe=False, verbose=False):
        """
        Fit a model using the variational Bayes mean field approximation.

        Parameters
        ----------
        mean : array_like
            Starting value for VB mean vector
        sd : array_like
            Starting value for VB standard deviation vector
        fit_method : str
            Algorithm for scipy.minimize
        minim_opts : dict
            Options passed to scipy.minimize
        scale_fe : bool
            If true, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.
        verbose : bool
            If True, print the gradient norm to the screen each time
            it is calculated.

        Notes
        -----
        The goal is to find a factored Gaussian approximation
        q1*q2*...  to the posterior distribution, approximately
        minimizing the KL divergence from the factored approximation
        to the actual posterior.  The KL divergence, or ELBO function
        has the form

            E* log p(y, fe, vcp, vc) - E* log q

        where E* is expectation with respect to the product of qj.

        References
        ----------
        Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A
        review for Statisticians
        https://arxiv.org/pdf/1601.00670.pdf
        """
        pass

class BayesMixedGLMResults:
    """
    Class to hold results from a Bayesian estimation of a Mixed GLM model.

    Attributes
    ----------
    fe_mean : array_like
        Posterior mean of the fixed effects coefficients.
    fe_sd : array_like
        Posterior standard deviation of the fixed effects coefficients
    vcp_mean : array_like
        Posterior mean of the logged variance component standard
        deviations.
    vcp_sd : array_like
        Posterior standard deviation of the logged variance component
        standard deviations.
    vc_mean : array_like
        Posterior mean of the random coefficients
    vc_sd : array_like
        Posterior standard deviation of the random coefficients
    """

    def __init__(self, model, params, cov_params, optim_retvals=None):
        self.model = model
        self.params = params
        self._cov_params = cov_params
        self.optim_retvals = optim_retvals
        self.fe_mean, self.vcp_mean, self.vc_mean = model._unpack(params)
        if cov_params.ndim == 2:
            cp = np.diag(cov_params)
        else:
            cp = cov_params
        self.fe_sd, self.vcp_sd, self.vc_sd = model._unpack(cp)
        self.fe_sd = np.sqrt(self.fe_sd)
        self.vcp_sd = np.sqrt(self.vcp_sd)
        self.vc_sd = np.sqrt(self.vc_sd)

    def random_effects(self, term=None):
        """
        Posterior mean and standard deviation of random effects.

        Parameters
        ----------
        term : int or None
            If None, results for all random effects are returned.  If
            an integer, returns results for a given set of random
            effects.  The value of `term` refers to an element of the
            `ident` vector, or to a position in the `vc_formulas`
            list.

        Returns
        -------
        Data frame of posterior means and posterior standard
        deviations of random effects.
        """
        pass

    def predict(self, exog=None, linear=False):
        """
        Return predicted values for the mean structure.

        Parameters
        ----------
        exog : array_like
            The design matrix for the mean structure.  If None,
            use the model's design matrix.
        linear : bool
            If True, returns the linear predictor, otherwise
            transform the linear predictor using the link function.

        Returns
        -------
        A one-dimensional array of fitted values.
        """
        pass

class BinomialBayesMixedGLM(_VariationalBayesMixedGLM, _BayesMixedGLM):
    __doc__ = _init_doc.format(example=_logit_example)

    def __init__(self, endog, exog, exog_vc, ident, vcp_p=1, fe_p=2, fep_names=None, vcp_names=None, vc_names=None):
        super(BinomialBayesMixedGLM, self).__init__(endog, exog, exog_vc=exog_vc, ident=ident, vcp_p=vcp_p, fe_p=fe_p, family=families.Binomial(), fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)
        if not np.all(np.unique(endog) == np.r_[0, 1]):
            msg = 'endog values must be 0 and 1, and not all identical'
            raise ValueError(msg)

    def vb_elbo(self, vb_mean, vb_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        """
        pass

    def vb_elbo_grad(self, vb_mean, vb_sd):
        """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """
        pass

class PoissonBayesMixedGLM(_VariationalBayesMixedGLM, _BayesMixedGLM):
    __doc__ = _init_doc.format(example=_poisson_example)

    def __init__(self, endog, exog, exog_vc, ident, vcp_p=1, fe_p=2, fep_names=None, vcp_names=None, vc_names=None):
        super(PoissonBayesMixedGLM, self).__init__(endog=endog, exog=exog, exog_vc=exog_vc, ident=ident, vcp_p=vcp_p, fe_p=fe_p, family=families.Poisson(), fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)

    def vb_elbo(self, vb_mean, vb_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        """
        pass

    def vb_elbo_grad(self, vb_mean, vb_sd):
        """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """
        pass