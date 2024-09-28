__all__ = ['ZeroInflatedPoisson', 'ZeroInflatedGeneralizedPoisson', 'ZeroInflatedNegativeBinomialP']
import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import DiscreteModel, CountModel, Poisson, Logit, CountResults, L1CountResults, Probit, _discrete_results_docs, _validate_l1_method, GeneralizedPoisson, NegativeBinomialP
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
_doc_zi_params = "\n    exog_infl : array_like or None\n        Explanatory variables for the binary inflation model, i.e. for\n        mixing probability model. If None, then a constant is used.\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n    inflation : {'logit', 'probit'}\n        The model for the zero inflation, either Logit (default) or Probit\n    "

class GenericZeroInflated(CountModel):
    __doc__ = '\n    Generic Zero Inflated Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, inflation='logit', exposure=None, missing='none', **kwargs):
        super(GenericZeroInflated, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        if exog_infl is None:
            self.k_inflate = 1
            self._no_exog_infl = True
            self.exog_infl = np.ones((endog.size, self.k_inflate), dtype=np.float64)
        else:
            self.exog_infl = exog_infl
            self.k_inflate = exog_infl.shape[1]
            self._no_exog_infl = False
        if len(exog.shape) == 1:
            self.k_exog = 1
        else:
            self.k_exog = exog.shape[1]
        self.infl = inflation
        if inflation == 'logit':
            self.model_infl = Logit(np.zeros(self.exog_infl.shape[0]), self.exog_infl)
            self._hessian_inflate = self._hessian_logit
        elif inflation == 'probit':
            self.model_infl = Probit(np.zeros(self.exog_infl.shape[0]), self.exog_infl)
            self._hessian_inflate = self._hessian_probit
        else:
            raise ValueError('inflation == %s, which is not handled' % inflation)
        self.inflation = inflation
        self.k_extra = self.k_inflate
        if len(self.exog) != len(self.exog_infl):
            raise ValueError('exog and exog_infl have different number ofobservation. `missing` handling is not supported')
        infl_names = ['inflate_%s' % i for i in self.model_infl.data.param_names]
        self.exog_names[:] = infl_names + list(self.exog_names)
        self.exog_infl = np.asarray(self.exog_infl, dtype=np.float64)
        self._init_keys.extend(['exog_infl', 'inflation'])
        self._null_drop_keys = ['exog_infl']

    def _get_exogs(self):
        """list of exogs, for internal use in post-estimation
        """
        pass

    def loglike(self, params):
        """
        Loglikelihood of Generic Zero Inflated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math:: \\ln L=\\sum_{y_{i}=0}\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\sum_{y_{i}>0}(\\ln(1-w_{i})+L_{main\\_model})
            where P - pdf of main model, L - loglike function of main model.
        """
        pass

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Zero Inflated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes for definition.

        Notes
        -----
        .. math:: \\ln L=\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\ln(1-w_{i})+L_{main\\_model}
            where P - pdf of main model, L - loglike function of main model.

        for observations :math:`i=1,...,n`
        """
        pass

    def score_obs(self, params):
        """
        Generic Zero Inflated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        pass

    def hessian(self, params):
        """
        Generic Zero Inflated model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        pass

    def predict(self, params, exog=None, exog_infl=None, exposure=None, offset=None, which='mean', y_values=None):
        """
        Predict expected response or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.
        which : str (optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x). This
              takes inflated zeros into account.
            - 'linear' : the linear predictor of the mean function.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'mean-main' : mean of the main count model
            - 'prob-main' : probability of selecting the main model.
                The probability of zero inflation is ``1 - prob-main``.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x)
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).

        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``
        """
        pass

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """NotImplemented
        """
        pass

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        """NotImplemented
        """
        pass

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        pass

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        pass

class ZeroInflatedPoisson(GenericZeroInflated):
    __doc__ = '\n    Poisson Zero Inflated Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', missing='none', **kwargs):
        super(ZeroInflatedPoisson, self).__init__(endog, exog, offset=offset, inflation=inflation, exog_infl=exog_infl, exposure=exposure, missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, offset=offset, exposure=exposure)
        self.distribution = zipoisson
        self.result_class = ZeroInflatedPoissonResults
        self.result_class_wrapper = ZeroInflatedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedPoissonResultsWrapper

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

    def get_distribution(self, params, exog=None, exog_infl=None, exposure=None, offset=None):
        """Get frozen instance of distribution based on predicted parameters.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor  of the mean
            function with coefficient equal to 1. If exposure is specified,
            then it will be logged by the method. The user does not need to
            log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.

        Returns
        -------
        Instance of frozen scipy distribution subclass.
        """
        pass

class ZeroInflatedGeneralizedPoisson(GenericZeroInflated):
    __doc__ = '\n    Zero Inflated Generalized Poisson Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    p : scalar\n        P denotes parametrizations for ZIGP regression.\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + 'p : float\n        dispersion power parameter for the GeneralizedPoisson model.  p=1 for\n        ZIGP-1 and p=2 for ZIGP-2. Default is p=2\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', p=2, missing='none', **kwargs):
        super(ZeroInflatedGeneralizedPoisson, self).__init__(endog, exog, offset=offset, inflation=inflation, exog_infl=exog_infl, exposure=exposure, missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog, offset=offset, exposure=exposure, p=p)
        self.distribution = zigenpoisson
        self.k_exog += 1
        self.k_extra += 1
        self.exog_names.append('alpha')
        self.result_class = ZeroInflatedGeneralizedPoissonResults
        self.result_class_wrapper = ZeroInflatedGeneralizedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedGeneralizedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedGeneralizedPoissonResultsWrapper

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

class ZeroInflatedNegativeBinomialP(GenericZeroInflated):
    __doc__ = '\n    Zero Inflated Generalized Negative Binomial Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    p : scalar\n        P denotes parametrizations for ZINB regression. p=1 for ZINB-1 and\n    p=2 for ZINB-2. Default is p=2\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + 'p : float\n        dispersion power parameter for the NegativeBinomialP model.  p=1 for\n        ZINB-1 and p=2 for ZINM-2. Default is p=2\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', p=2, missing='none', **kwargs):
        super(ZeroInflatedNegativeBinomialP, self).__init__(endog, exog, offset=offset, inflation=inflation, exog_infl=exog_infl, exposure=exposure, missing=missing, **kwargs)
        self.model_main = NegativeBinomialP(self.endog, self.exog, offset=offset, exposure=exposure, p=p)
        self.distribution = zinegbin
        self.k_exog += 1
        self.k_extra += 1
        self.exog_names.append('alpha')
        self.result_class = ZeroInflatedNegativeBinomialResults
        self.result_class_wrapper = ZeroInflatedNegativeBinomialResultsWrapper
        self.result_class_reg = L1ZeroInflatedNegativeBinomialResults
        self.result_class_reg_wrapper = L1ZeroInflatedNegativeBinomialResultsWrapper

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

class ZeroInflatedResults(CountResults):

    def get_influence(self):
        """
        Influence and outlier measures

        See notes section for influence measures that do not apply for
        zero inflated models.

        Returns
        -------
        MLEInfluence
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        ZeroInflated models have functions that are not differentiable
        with respect to sample endog if endog=0. This means that generalized
        leverage cannot be computed in the usual definition.

        Currently, both the generalized leverage, in `hat_matrix_diag`
        attribute and studetized residuals are not available. In the influence
        plot generalized leverage is replaced by a hat matrix diagonal that
        only takes combined exog into account, computed in the same way as
        for OLS. This is a measure for exog outliers but does not take
        specific features of the model into account.
        """
        pass

class ZeroInflatedPoissonResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Zero Inflated Poisson', 'extra_attr': ''}

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        pass

class L1ZeroInflatedPoissonResults(L1CountResults, ZeroInflatedPoissonResults):
    pass

class ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedPoissonResultsWrapper, ZeroInflatedPoissonResults)

class L1ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedPoissonResultsWrapper, L1ZeroInflatedPoissonResults)

class ZeroInflatedGeneralizedPoissonResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Zero Inflated Generalized Poisson', 'extra_attr': ''}

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        pass

class L1ZeroInflatedGeneralizedPoissonResults(L1CountResults, ZeroInflatedGeneralizedPoissonResults):
    pass

class ZeroInflatedGeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedGeneralizedPoissonResultsWrapper, ZeroInflatedGeneralizedPoissonResults)

class L1ZeroInflatedGeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedGeneralizedPoissonResultsWrapper, L1ZeroInflatedGeneralizedPoissonResults)

class ZeroInflatedNegativeBinomialResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Zero Inflated Generalized Negative Binomial', 'extra_attr': ''}

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        pass

class L1ZeroInflatedNegativeBinomialResults(L1CountResults, ZeroInflatedNegativeBinomialResults):
    pass

class ZeroInflatedNegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedNegativeBinomialResultsWrapper, ZeroInflatedNegativeBinomialResults)

class L1ZeroInflatedNegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedNegativeBinomialResultsWrapper, L1ZeroInflatedNegativeBinomialResults)