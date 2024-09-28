from __future__ import division
__all__ = ['TruncatedLFPoisson', 'TruncatedLFNegativeBinomialP', 'HurdleCountModel']
import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import truncatedpoisson, truncatednegbin
from statsmodels.discrete.discrete_model import DiscreteModel, CountModel, CountResults, L1CountResults, Poisson, NegativeBinomialP, GeneralizedPoisson, _discrete_results_docs
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy

class TruncatedLFGeneric(CountModel):
    __doc__ = '\n    Generic Truncated model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, truncation=0, offset=None, exposure=None, missing='none', **kwargs):
        super(TruncatedLFGeneric, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        mask = self.endog > truncation
        self.exog = self.exog[mask]
        self.endog = self.endog[mask]
        if offset is not None:
            self.offset = self.offset[mask]
        if exposure is not None:
            self.exposure = self.exposure[mask]
        self.trunc = truncation
        self.truncation = truncation
        self._init_keys.extend(['truncation'])
        self._null_drop_keys = []

    def loglike(self, params):
        """
        Loglikelihood of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----

        """
        pass

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Truncated model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----

        """
        pass

    def score_obs(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        pass

    def score(self, params):
        """
        Generic Truncated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        pass
    fit.__doc__ = DiscreteModel.fit.__doc__
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Truncated model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
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

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        """
        Predict response variable or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
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

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob-trunc' : probability of truncation. This is the probability
              of observing a zero count implied
              by the truncation model.
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).
              The probabilities in the truncated region are zero.
            - 'prob-base' : probabilities for untruncated base distribution.
              The probabilities are for each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).


        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        predicted values

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        pass

class TruncatedLFPoisson(TruncatedLFGeneric):
    __doc__ = '\n    Truncated Poisson model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, truncation=0, missing='none', **kwargs):
        super(TruncatedLFPoisson, self).__init__(endog, exog, offset=offset, exposure=exposure, truncation=truncation, missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, exposure=getattr(self, 'exposure', None), offset=getattr(self, 'offset', None))
        self.model_dist = truncatedpoisson
        self.result_class = TruncatedLFPoissonResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _predict_mom_trunc0(self, params, mu):
        """Predict mean and variance of zero-truncated distribution.

        experimental api, will likely be replaced by other methods

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

class TruncatedLFNegativeBinomialP(TruncatedLFGeneric):
    __doc__ = '\n    Truncated Generalized Negative Binomial model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, truncation=0, p=2, missing='none', **kwargs):
        super(TruncatedLFNegativeBinomialP, self).__init__(endog, exog, offset=offset, exposure=exposure, truncation=truncation, missing=missing, **kwargs)
        self.model_main = NegativeBinomialP(self.endog, self.exog, exposure=getattr(self, 'exposure', None), offset=getattr(self, 'offset', None), p=p)
        self.k_extra = self.model_main.k_extra
        self.exog_names.extend(self.model_main.exog_names[-self.k_extra:])
        self.model_dist = truncatednegbin
        self.result_class = TruncatedNegativeBinomialResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _predict_mom_trunc0(self, params, mu):
        """Predict mean and variance of zero-truncated distribution.

        experimental api, will likely be replaced by other methods

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.

        Returns
        -------
        Predicted conditional variance.
        """
        pass

class TruncatedLFGeneralizedPoisson(TruncatedLFGeneric):
    __doc__ = '\n    Truncated Generalized Poisson model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, truncation=0, p=2, missing='none', **kwargs):
        super(TruncatedLFGeneralizedPoisson, self).__init__(endog, exog, offset=offset, exposure=exposure, truncation=truncation, missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog, exposure=getattr(self, 'exposure', None), offset=getattr(self, 'offset', None), p=p)
        self.k_extra = self.model_main.k_extra
        self.exog_names.extend(self.model_main.exog_names[-self.k_extra:])
        self.model_dist = None
        self.result_class = TruncatedNegativeBinomialResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

class _RCensoredGeneric(CountModel):
    __doc__ = '\n    Generic right Censored model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none', **kwargs):
        self.zero_idx = np.nonzero(endog == 0)[0]
        self.nonzero_idx = np.nonzero(endog)[0]
        super(_RCensoredGeneric, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)

    def loglike(self, params):
        """
        Loglikelihood of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----

        """
        pass

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----

        """
        pass

    def score_obs(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        pass

    def score(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        pass
    fit.__doc__ = DiscreteModel.fit.__doc__
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Censored model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
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

class _RCensoredPoisson(_RCensoredGeneric):
    __doc__ = '\n    Censored Poisson model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none', **kwargs):
        super(_RCensoredPoisson, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.model_main = Poisson(np.zeros_like(self.endog), self.exog)
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

class _RCensoredGeneralizedPoisson(_RCensoredGeneric):
    __doc__ = '\n    Censored Generalized Poisson model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, p=2, exposure=None, missing='none', **kwargs):
        super(_RCensoredGeneralizedPoisson, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(np.zeros_like(self.endog), self.exog)
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

class _RCensoredNegativeBinomialP(_RCensoredGeneric):
    __doc__ = '\n    Censored Negative Binomial model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, p=2, exposure=None, missing='none', **kwargs):
        super(_RCensoredNegativeBinomialP, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.model_main = NegativeBinomialP(np.zeros_like(self.endog), self.exog, p=p)
        self.model_dist = None
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

class _RCensored(_RCensoredGeneric):
    __doc__ = '\n    Censored model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, model=Poisson, distribution=truncatedpoisson, offset=None, exposure=None, missing='none', **kwargs):
        super(_RCensored, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.model_main = model(np.zeros_like(self.endog), self.exog)
        self.model_dist = distribution
        self.k_extra = k_extra = self.model_main.k_extra
        if k_extra > 0:
            self.exog_names.extend(self.model_main.exog_names[-k_extra:])
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _prob_nonzero(self, mu, params):
        """Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
        pass

class HurdleCountModel(CountModel):
    __doc__ = "\n    Hurdle model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    dist : string\n        Log-likelihood type of count model family. 'poisson' or 'negbin'\n    zerodist : string\n        Log-likelihood type of zero hurdle model family. 'poisson', 'negbin'\n    p : scalar\n        Define parameterization for count model.\n        Used when dist='negbin'.\n    pzero : scalar\n        Define parameterization parameter zero hurdle model family.\n        Used when zerodist='negbin'.\n    " % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    Notes\n    -----\n    The parameters in the NegativeBinomial zero model are not identified if\n    the predicted mean is constant. If there is no or only little variation in\n    the predicted mean, then convergence might fail, hessian might not be\n    invertible or parameter estimates will have large standard errors.\n\n    References\n    ----------\n    not yet\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, dist='poisson', zerodist='poisson', p=2, pzero=2, exposure=None, missing='none', **kwargs):
        if offset is not None or exposure is not None:
            msg = 'Offset and exposure are not yet implemented'
            raise NotImplementedError(msg)
        super(HurdleCountModel, self).__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.k_extra1 = 0
        self.k_extra2 = 0
        self._initialize(dist, zerodist, p, pzero)
        self.result_class = HurdleCountResults
        self.result_class_wrapper = HurdleCountResultsWrapper
        self.result_class_reg = L1HurdleCountResults
        self.result_class_reg_wrapper = L1HurdleCountResultsWrapper

    def loglike(self, params):
        """
        Loglikelihood of Generic Hurdle model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----

        """
        pass
    fit.__doc__ = DiscreteModel.fit.__doc__

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        """
        Predict response variable or other statistic given exogenous variables.

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

            - 'mean' : the conditional expectation of endog E(y | x)
            - 'mean-main' : mean parameter of truncated count model.
              Note, this is not the mean of the truncated distribution.
            - 'linear' : the linear predictor of the truncated count model.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'prob-main' : probability of selecting the main model which is
              the probability of observing a nonzero count P(y > 0 | x).
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x).
              This is equal to is ``1 - prob-main``
            - 'prob-trunc' : probability of truncation of the truncated count
              model. This is the probability of observing a zero count implied
              by the truncation model.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).

        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``

        Returns
        -------
        predicted values

        Notes
        -----
        'prob-zero' / 'prob-trunc' is the ratio of probabilities of observing
        a zero count between hurdle model and the truncated count model.
        If this ratio is larger than one, then the hurdle model has an inflated
        number of zeros compared to the count model. If it is smaller than one,
        then the number of zeros is deflated.
        """
        pass

class TruncatedLFGenericResults(CountResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Generic Truncated', 'extra_attr': ''}

class TruncatedLFPoissonResults(TruncatedLFGenericResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Truncated Poisson', 'extra_attr': ''}

class TruncatedNegativeBinomialResults(TruncatedLFGenericResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Truncated Negative Binomial', 'extra_attr': ''}

class L1TruncatedLFGenericResults(L1CountResults, TruncatedLFGenericResults):
    pass

class TruncatedLFGenericResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(TruncatedLFGenericResultsWrapper, TruncatedLFGenericResults)

class L1TruncatedLFGenericResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1TruncatedLFGenericResultsWrapper, L1TruncatedLFGenericResults)

class HurdleCountResults(CountResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Hurdle model', 'extra_attr': ''}

    def __init__(self, model, mlefit, results_zero, results_count, cov_type='nonrobust', cov_kwds=None, use_t=None):
        super(HurdleCountResults, self).__init__(model, mlefit, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        self.results_zero = results_zero
        self.results_count = results_count
        self.df_resid = self.model.endog.shape[0] - len(self.params)

class L1HurdleCountResults(L1CountResults, HurdleCountResults):
    pass

class HurdleCountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(HurdleCountResultsWrapper, HurdleCountResults)

class L1HurdleCountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1HurdleCountResultsWrapper, L1HurdleCountResults)