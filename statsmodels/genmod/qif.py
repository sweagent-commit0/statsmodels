import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly

class QIFCovariance:
    """
    A covariance model for quadratic inference function regression.

    The mat method returns a basis matrix B such that the inverse
    of the working covariance lies in the linear span of the
    basis matrices.

    Subclasses should set the number of basis matrices `num_terms`,
    so that `mat(d, j)` for j=0, ..., num_terms-1 gives the basis
    of dimension d.`
    """

    def mat(self, dim, term):
        """
        Returns the term'th basis matrix, which is a dim x dim
        matrix.
        """
        pass

class QIFIndependence(QIFCovariance):
    """
    Independent working covariance for QIF regression.  This covariance
    model gives identical results to GEE with the independence working
    covariance.  When using QIFIndependence as the working covariance,
    the QIF value will be zero, and cannot be used for chi^2 testing, or
    for model selection using AIC, BIC, etc.
    """

    def __init__(self):
        self.num_terms = 1

class QIFExchangeable(QIFCovariance):
    """
    Exchangeable working covariance for QIF regression.
    """

    def __init__(self):
        self.num_terms = 2

class QIFAutoregressive(QIFCovariance):
    """
    Autoregressive working covariance for QIF regression.
    """

    def __init__(self):
        self.num_terms = 3

class QIF(base.Model):
    """
    Fit a regression model using quadratic inference functions (QIF).

    QIF is an alternative to GEE that can be more efficient, and that
    offers different approaches for model selection and inference.

    Parameters
    ----------
    endog : array_like
        The dependent variables of the regression.
    exog : array_like
        The independent variables of the regression.
    groups : array_like
        Labels indicating which group each observation belongs to.
        Observations in different groups should be independent.
    family : genmod family
        An instance of a GLM family.\x7f
    cov_struct : QIFCovariance instance
        An instance of a QIFCovariance.

    References
    ----------
    A. Qu, B. Lindsay, B. Li (2000).  Improving Generalized Estimating
    Equations using Quadratic Inference Functions, Biometrika 87:4.
    www.jstor.org/stable/2673612
    """

    def __init__(self, endog, exog, groups, family=None, cov_struct=None, missing='none', **kwargs):
        if family is None:
            family = families.Gaussian()
        elif not issubclass(family.__class__, families.Family):
            raise ValueError('QIF: `family` must be a genmod family instance')
        self.family = family
        self._fit_history = defaultdict(list)
        if cov_struct is None:
            cov_struct = QIFIndependence()
        elif not isinstance(cov_struct, QIFCovariance):
            raise ValueError('QIF: `cov_struct` must be a QIFCovariance instance')
        self.cov_struct = cov_struct
        groups = np.asarray(groups)
        super(QIF, self).__init__(endog, exog, groups=groups, missing=missing, **kwargs)
        self.group_names = list(set(groups))
        self.nobs = len(self.endog)
        groups_ix = defaultdict(list)
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self.groups_ix = [groups_ix[na] for na in self.group_names]
        self._check_args(groups)

    def objective(self, params):
        """
        Calculate the gradient of the QIF objective function.

        Parameters
        ----------
        params : array_like
            The model parameters at which the gradient is evaluated.

        Returns
        -------
        grad : array_like
            The gradient vector of the QIF objective function.
        gn_deriv : array_like
            The gradients of each estimating equation with
            respect to the parameter.
        """
        pass

    def estimate_scale(self, params):
        """
        Estimate the dispersion/scale.

        The scale parameter for binomial and Poisson families is
        fixed at 1, otherwise it is estimated from the data.
        """
        pass

    @classmethod
    def from_formula(cls, formula, groups, data, subset=None, *args, **kwargs):
        """
        Create a QIF model instance from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        groups : array_like or string
            Array of grouping labels.  If a string, this is the name
            of a variable in `data` that contains the grouping labels.
        data : array_like
            The data for the model.
        subset : array_like
            An array_like object of booleans, integers, or index
            values that indicate the subset of the data to used when
            fitting the model.

        Returns
        -------
        model : QIF model instance
        """
        pass

    def fit(self, maxiter=100, start_params=None, tol=1e-06, gtol=0.0001, ddof_scale=None):
        """
        Fit a GLM to correlated data using QIF.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        start_params : array_like, optional
            Starting values
        tol : float
            Convergence threshold for difference of successive
            estimates.
        gtol : float
            Convergence threshold for gradient.
        ddof_scale : int, optional
            Degrees of freedom for the scale parameter

        Returns
        -------
        QIFResults object
        """
        pass

class QIFResults(base.LikelihoodModelResults):
    """Results class for QIF Regression"""

    def __init__(self, model, params, cov_params, scale, use_t=False, **kwds):
        super(QIFResults, self).__init__(model, params, normalized_cov_params=cov_params, scale=scale)
        self.qif, _, _, _, _ = self.model.objective(params)

    @cache_readonly
    def aic(self):
        """
        An AIC-like statistic for models fit using QIF.
        """
        pass

    @cache_readonly
    def bic(self):
        """
        A BIC-like statistic for models fit using QIF.
        """
        pass

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values from the model.
        """
        pass

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the QIF regression results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
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
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        pass

class QIFResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(QIFResultsWrapper, QIFResults)