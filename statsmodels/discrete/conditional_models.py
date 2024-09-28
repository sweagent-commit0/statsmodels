"""
Conditional logistic, Poisson, and multinomial logit regression
"""
import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import MultinomialResults, MultinomialResultsWrapper
import collections
import warnings
import itertools

class _ConditionalModel(base.LikelihoodModel):

    def __init__(self, endog, exog, missing='none', **kwargs):
        if 'groups' not in kwargs:
            raise ValueError("'groups' is a required argument")
        groups = kwargs['groups']
        if groups.size != endog.size:
            msg = "'endog' and 'groups' should have the same dimensions"
            raise ValueError(msg)
        if exog.shape[0] != endog.size:
            msg = "The leading dimension of 'exog' should equal the length of 'endog'"
            raise ValueError(msg)
        super(_ConditionalModel, self).__init__(endog, exog, missing=missing, **kwargs)
        if self.data.const_idx is not None:
            msg = 'Conditional models should not have an intercept in the ' + 'design matrix'
            raise ValueError(msg)
        exog = self.exog
        self.k_params = exog.shape[1]
        row_ix = {}
        for i, g in enumerate(groups):
            if g not in row_ix:
                row_ix[g] = []
            row_ix[g].append(i)
        endog, exog = (np.asarray(endog), np.asarray(exog))
        offset = kwargs.get('offset')
        self._endog_grp = []
        self._exog_grp = []
        self._groupsize = []
        if offset is not None:
            offset = np.asarray(offset)
            self._offset_grp = []
        self._offset = []
        self._sumy = []
        self.nobs = 0
        drops = [0, 0]
        for g, ix in row_ix.items():
            y = endog[ix].flat
            if np.std(y) == 0:
                drops[0] += 1
                drops[1] += len(y)
                continue
            self.nobs += len(y)
            self._endog_grp.append(y)
            if offset is not None:
                self._offset_grp.append(offset[ix])
            self._groupsize.append(len(y))
            self._exog_grp.append(exog[ix, :])
            self._sumy.append(np.sum(y))
        if drops[0] > 0:
            msg = ('Dropped %d groups and %d observations for having ' + 'no within-group variance') % tuple(drops)
            warnings.warn(msg)
        if offset is not None:
            self._endofs = []
            for k, ofs in enumerate(self._offset_grp):
                self._endofs.append(np.dot(self._endog_grp[k], ofs))
        self._n_groups = len(self._endog_grp)
        self._xy = []
        self._n1 = []
        for g in range(self._n_groups):
            self._xy.append(np.dot(self._endog_grp[g], self._exog_grp[g]))
            self._n1.append(np.sum(self._endog_grp[g]))

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword argument that are used when fitting the model.

        Returns
        -------
        Results
            A results instance.
        """
        pass

class ConditionalLogit(_ConditionalModel):
    """
    Fit a conditional logistic regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable, must contain only 0 and 1.
    exog : array_like
        The array of covariates.  Do not include an intercept
        in this array.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):
        super(ConditionalLogit, self).__init__(endog, exog, missing=missing, **kwargs)
        if np.any(np.unique(self.endog) != np.r_[0, 1]):
            msg = 'endog must be coded as 0, 1'
            raise ValueError(msg)
        self.K = self.exog.shape[1]

class ConditionalPoisson(_ConditionalModel):
    """
    Fit a conditional Poisson regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable
    exog : array_like
        The covariates
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

class ConditionalResults(base.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params, scale):
        super(ConditionalResults, self).__init__(model, params, normalized_cov_params=normalized_cov_params, scale=scale)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the fitted model.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """
        pass

class ConditionalMNLogit(_ConditionalModel):
    """
    Fit a conditional multinomial logit model to grouped data.

    Parameters
    ----------
    endog : array_like
        The dependent variable, must be integer-valued, coded
        0, 1, ..., c-1, where c is the number of response
        categories.
    exog : array_like
        The independent variables.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.

    Notes
    -----
    Equivalent to femlogit in Stata.

    References
    ----------
    Gary Chamberlain (1980).  Analysis of covariance with qualitative
    data. The Review of Economic Studies.  Vol. 47, No. 1, pp. 225-238.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):
        super(ConditionalMNLogit, self).__init__(endog, exog, missing=missing, **kwargs)
        self.endog = self.endog.astype(int)
        self.k_cat = self.endog.max() + 1
        self.df_model = (self.k_cat - 1) * self.exog.shape[1]
        self.df_resid = self.nobs - self.df_model
        self._ynames_map = {j: str(j) for j in range(self.k_cat)}
        self.J = self.k_cat
        self.K = self.exog.shape[1]
        if self.endog.min() < 0:
            msg = 'endog may not contain negative values'
            raise ValueError(msg)
        grx = collections.defaultdict(list)
        for k, v in enumerate(self.groups):
            grx[v].append(k)
        self._group_labels = list(grx.keys())
        self._group_labels.sort()
        self._grp_ix = [grx[k] for k in self._group_labels]

class ConditionalResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ConditionalResultsWrapper, ConditionalResults)