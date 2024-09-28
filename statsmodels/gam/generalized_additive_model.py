"""
Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

created on 08/07/2015
"""
from collections.abc import Iterable
import copy
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults, GLMResultsWrapper, _check_convergence
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ValueWarning
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import MultivariateGAMCVPath
from statsmodels.gam.gam_cross_validation.cross_validators import KFold

def _transform_predict_exog(model, exog, design_info=None):
    """transform exog for predict using design_info

    Note: this is copied from base.model.Results.predict and converted to
    standalone function with additional options.
    """
    pass

class GLMGamResults(GLMResults):
    """Results class for generalized additive models, GAM.

    This inherits from GLMResults.

    Warning: some inherited methods might not correctly take account of the
    penalization

    GLMGamResults inherits from GLMResults
    All methods related to the loglikelihood function return the penalized
    values.

    Attributes
    ----------

    edf
        list of effective degrees of freedom for each column of the design
        matrix.
    hat_matrix_diag
        diagonal of hat matrix
    gcv
        generalized cross-validation criterion computed as
        ``gcv = scale / (1. - hat_matrix_trace / self.nobs)**2``
    cv
        cross-validation criterion computed as
        ``cv = ((resid_pearson / (1 - hat_matrix_diag))**2).sum() / nobs``

    Notes
    -----
    status: experimental
    """

    def __init__(self, model, params, normalized_cov_params, scale, **kwds):
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        edf = self.edf.sum()
        self.df_model = edf - 1
        self.df_resid = self.model.endog.shape[0] - edf
        self.model.df_model = self.df_model
        self.model.df_resid = self.df_resid
        mu = self.fittedvalues
        self.scale = scale = self.model.estimate_scale(mu)
        super(GLMGamResults, self).__init__(model, params, normalized_cov_params, scale, **kwds)

    def _tranform_predict_exog(self, exog=None, exog_smooth=None, transform=True):
        """Transform original explanatory variables for prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is False, then ``exog`` is returned unchanged and
            ``x`` is ignored. It is assumed that exog contains the full
            design matrix for the predict observations.
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.

        Returns
        -------
        exog_transformed : ndarray
            design matrix for the prediction
        """
        pass

    def predict(self, exog=None, exog_smooth=None, transform=True, **kwargs):
        """"
        compute prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``exog``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction : ndarray, pandas.Series or pandas.DataFrame
            predicted values
        """
        pass

    def get_prediction(self, exog=None, exog_smooth=None, transform=True, **kwargs):
        """compute prediction results

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction_results : generalized_linear_model.PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary tables for the prediction of the mean and of new
            observations.
        """
        pass

    def partial_values(self, smooth_index, include_constant=True):
        """contribution of a smooth term to the linear prediction

        Warning: This will be replaced by a predict method

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.

        Returns
        -------
        predicted : nd_array
            predicted value of linear term.
            This is not the expected response if the link function is not
            linear.
        se_pred : nd_array
            standard error of linear prediction
        """
        pass

    def plot_partial(self, smooth_index, plot_se=True, cpr=False, include_constant=True, ax=None):
        """plot the contribution of a smooth term to the linear prediction

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        plot_se : bool
            If plot_se is true, then the confidence interval for the linear
            prediction will be added to the plot.
        cpr : bool
            If cpr (component plus residual) is true, then a scatter plot of
            the partial working residuals will be added to the plot.
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.
        ax : None or matplotlib axis instance
           If ax is not None, then the plot will be added to it.

        Returns
        -------
        Figure
            If `ax` is None, the created figure. Otherwise, the Figure to which
            `ax` is connected.
        """
        pass

    def test_significance(self, smooth_index):
        """hypothesis test that a smooth component is zero.

        This calls `wald_test` to compute the hypothesis test, but uses
        effective degrees of freedom.

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms

        Returns
        -------
        wald_test : ContrastResults instance
            the results instance created by `wald_test`
        """
        pass

    def get_hat_matrix_diag(self, observed=True, _axis=1):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.
            This is only relevant for models that implement both observed
            and expected Hessian, which is currently only GLM. Other
            models only use the observed Hessian.
        _axis : int
            This is mainly for internal use. By default it returns the usual
            diagonal of the hat matrix. If _axis is zero, then the result
            corresponds to the effective degrees of freedom, ``edf`` for each
            column of exog.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        pass

class GLMGamResultsWrapper(GLMResultsWrapper):
    pass
wrap.populate_wrapper(GLMGamResultsWrapper, GLMGamResults)

class GLMGam(PenalizedMixin, GLM):
    """
    Generalized Additive Models (GAM)

    This inherits from `GLM`.

    Warning: Not all inherited methods might take correctly account of the
    penalization. Not all options including offset and exposure have been
    verified yet.

    Parameters
    ----------
    endog : array_like
        The response variable.
    exog : array_like or None
        This explanatory variables are treated as linear. The model in this
        case is a partial linear model.
    smoother : instance of additive smoother class
        Examples of smoother instances include Bsplines or CyclicCubicSplines.
    alpha : float or list of floats
        Penalization weights for smooth terms. The length of the list needs
        to be the same as the number of smooth terms in the ``smoother``.
    family : instance of GLM family
        See GLM.
    offset : None or array_like
        See GLM.
    exposure : None or array_like
        See GLM.
    missing : 'none'
        Missing value handling is not supported in this class.
    **kwargs
        Extra keywords are used in call to the super classes.

    Notes
    -----
    Status: experimental. This has full unit test coverage for the core
    results with Gaussian and Poisson (without offset and exposure). Other
    options and additional results might not be correctly supported yet.
    (Binomial with counts, i.e. with n_trials, is most likely wrong in pirls.
    User specified var or freq weights are most likely also not correct for
    all results.)
    """
    _results_class = GLMGamResults
    _results_class_wrapper = GLMGamResultsWrapper

    def __init__(self, endog, exog=None, smoother=None, alpha=0, family=None, offset=None, exposure=None, missing='none', **kwargs):
        hasconst = kwargs.get('hasconst', None)
        xnames_linear = None
        if hasattr(exog, 'design_info'):
            self.design_info_linear = exog.design_info
            xnames_linear = self.design_info_linear.column_names
        is_pandas = _is_using_pandas(exog, None)
        self.data_linear = self._handle_data(endog, exog, missing, hasconst)
        if xnames_linear is None:
            xnames_linear = self.data_linear.xnames
        if exog is not None:
            exog_linear = self.data_linear.exog
            k_exog_linear = exog_linear.shape[1]
        else:
            exog_linear = None
            k_exog_linear = 0
        self.k_exog_linear = k_exog_linear
        self.exog_linear = exog_linear
        self.smoother = smoother
        self.k_smooths = smoother.k_variables
        self.alpha = self._check_alpha(alpha)
        penal = MultivariateGamPenalty(smoother, alpha=self.alpha, start_idx=k_exog_linear)
        kwargs.pop('penal', None)
        if exog_linear is not None:
            exog = np.column_stack((exog_linear, smoother.basis))
        else:
            exog = smoother.basis
        if xnames_linear is None:
            xnames_linear = []
        xnames = xnames_linear + self.smoother.col_names
        if is_pandas and exog_linear is not None:
            exog = pd.DataFrame(exog, index=self.data_linear.row_labels, columns=xnames)
        super(GLMGam, self).__init__(endog, exog=exog, family=family, offset=offset, exposure=exposure, penal=penal, missing=missing, **kwargs)
        if not is_pandas:
            self.exog_names[:] = xnames
        if hasattr(self.data, 'design_info'):
            del self.data.design_info
        if hasattr(self, 'formula'):
            self.formula_linear = self.formula
            self.formula = None
            del self.formula

    def _check_alpha(self, alpha):
        """check and convert alpha to required list format

        Parameters
        ----------
        alpha : scalar, list or array_like
            penalization weight

        Returns
        -------
        alpha : list
            penalization weight, list with length equal to the number of
            smooth terms
        """
        pass

    def fit(self, start_params=None, maxiter=1000, method='pirls', tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, full_output=True, disp=False, max_start_irls=3, **kwargs):
        """estimate parameters and create instance of GLMGamResults class

        Parameters
        ----------
        most parameters are the same as for GLM
        method : optimization method
            The special optimization method is "pirls" which uses a penalized
            version of IRLS. Other methods are gradient optimizers as used in
            base.model.LikelihoodModel.

        Returns
        -------
        res : instance of wrapped GLMGamResults
        """
        pass

    def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):
        """fit model with penalized reweighted least squares
        """
        pass

    def select_penweight(self, criterion='aic', start_params=None, start_model_params=None, method='basinhopping', **fit_kwds):
        """find alpha by minimizing results criterion

        The objective for the minimization can be results attributes like
        ``gcv``, ``aic`` or ``bic`` where the latter are based on effective
        degrees of freedom.

        Warning: In many case the optimization might converge to a local
        optimum or near optimum. Different start_params or using a global
        optimizer is recommended, default is basinhopping.

        Parameters
        ----------
        criterion='aic'
            name of results attribute to be minimized.
            Default is 'aic', other options are 'gcv', 'cv' or 'bic'.
        start_params : None or array
            starting parameters for alpha in the penalization weight
            minimization. The parameters are internally exponentiated and
            the minimization is with respect to ``exp(alpha)``
        start_model_params : None or array
            starting parameter for the ``model._fit_pirls``.
        method : 'basinhopping', 'nm' or 'minimize'
            'basinhopping' and 'nm' directly use the underlying scipy.optimize
            functions `basinhopping` and `fmin`. 'minimize' provides access
            to the high level interface, `scipy.optimize.minimize`.
        fit_kwds : keyword arguments
            additional keyword arguments will be used in the call to the
            scipy optimizer. Which keywords are supported depends on the
            scipy optimization function.

        Returns
        -------
        alpha : ndarray
            penalization parameter found by minimizing the criterion.
            Note that this can be only a local (near) optimum.
        fit_res : tuple
            results returned by the scipy optimization routine. The
            parameters in the optimization problem are `log(alpha)`
        history : dict
            history of calls to pirls and contains alpha, the fit
            criterion and the parameters to which pirls converged to for the
            given alpha.

        Notes
        -----
        In the test cases Nelder-Mead and bfgs often converge to local optima,
        see also https://github.com/statsmodels/statsmodels/issues/5381.

        This does not use any analytical derivatives for the criterion
        minimization.

        Status: experimental, It is possible that defaults change if there
        is a better way to find a global optimum. API (e.g. type of return)
        might also change.
        """
        pass

    def select_penweight_kfold(self, alphas=None, cv_iterator=None, cost=None, k_folds=5, k_grid=11):
        """find alphas by k-fold cross-validation

        Warning: This estimates ``k_folds`` models for each point in the
            grid of alphas.

        Parameters
        ----------
        alphas : None or list of arrays
        cv_iterator : instance
            instance of a cross-validation iterator, by default this is a
            KFold instance
        cost : function
            default is mean squared error. The cost function to evaluate the
            prediction error for the left out sample. This should take two
            arrays as argument and return one float.
        k_folds : int
            number of folds if default Kfold iterator is used.
            This is ignored if ``cv_iterator`` is not None.

        Returns
        -------
        alpha_cv : list of float
            Best alpha in grid according to cross-validation
        res_cv : instance of MultivariateGAMCVPath
            The instance was used for cross-validation and holds the results

        Notes
        -----
        The default alphas are defined as
        ``alphas = [np.logspace(0, 7, k_grid) for _ in range(k_smooths)]``
        """
        pass

class LogitGam(PenalizedMixin, Logit):
    """Generalized Additive model for discrete Logit

    This subclasses discrete_model Logit.

    Warning: not all inherited methods might take correctly account of the
    penalization

    not verified yet.
    """

    def __init__(self, endog, smoother, alpha, *args, **kwargs):
        if not isinstance(alpha, Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers))
        self.smoother = smoother
        self.alpha = alpha
        self.pen_weight = 1
        penal = MultivariateGamPenalty(smoother, alpha=alpha)
        super(LogitGam, self).__init__(endog, smoother.basis, *args, penal=penal, **kwargs)

def penalized_wls(endog, exog, penalty_matrix, weights):
    """weighted least squares with quadratic penalty

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization. Note, the penalty_matrix
        is multiplied by two to match non-pirls fitting methods.
    weights : ndarray
        weights for WLS

    Returns
    -------
    results : Results instance of WLS
    """
    pass

def make_augmented_matrix(endog, exog, penalty_matrix, weights):
    """augment endog, exog and weights with stochastic restriction matrix

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization
    weights : ndarray
        weights for WLS

    Returns
    -------
    endog_aug : ndarray
        augmented response variable
    exog_aug : ndarray
        augmented design matrix
    weights_aug : ndarray
        augmented weights for WLS
    """
    pass