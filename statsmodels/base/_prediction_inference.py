"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
import pandas as pd

class PredictionResultsBase:
    """Based class for get_prediction results
    """

    def __init__(self, predicted, var_pred, func=None, deriv=None, df=None, dist=None, row_labels=None, **kwds):
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def t_test(self, value=0, alternative='two-sided'):
        """z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        """
        pass

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        pass

    def conf_int(self, *, alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            Ignored in base class, only for compatibility, consistent signature
            with subclasses

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        pass

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns 'predicted', 'se', 'ci_lower', 'ci_upper'
        """
        pass

class PredictionResultsMonotonic(PredictionResultsBase):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None, func=None, deriv=None, df=None, dist=None, row_labels=None):
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        pass

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        pass

class PredictionResultsDelta(PredictionResultsBase):
    """Prediction results based on delta method
    """

    def __init__(self, results_delta, **kwds):
        predicted = results_delta.predicted()
        var_pred = results_delta.var()
        super().__init__(predicted, var_pred, **kwds)

class PredictionResultsMean(PredictionResultsBase):
    """Prediction results for GLM.

    This results class is used for backwards compatibility for
    `get_prediction` with GLM. The new PredictionResults classes dropped the
    `_mean` post fix in the attribute names.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid=None, df=None, dist=None, row_labels=None, linpred=None, link=None):
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        pass

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns
        'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'.
        """
        pass

def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    """Prepare or transform exog for prediction

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    exog : ndarray
        Prediction exog
    row_labels : list of str
        Labels or pandas index for rows of prediction
    """
    pass

def get_prediction_glm(self, exog=None, transform=True, row_labels=None, linpred=None, link=None, pred_kwds=None):
    """
    Compute prediction results for GLM compatible models.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    linpred : linear prediction instance
        Instance of linear prediction results used for confidence intervals
        based on endpoint transformation.
    link : instance of link function
        If no link function is provided, then the `model.family.link` is used.
    pred_kwds : dict
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models. See the predict method of the
        model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """
    pass

def get_prediction_linear(self, exog=None, transform=True, row_labels=None, pred_kwds=None, index=None):
    """
    Compute prediction results for linear prediction.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """
    pass

def get_prediction_monotonic(self, exog=None, transform=True, row_labels=None, link=None, pred_kwds=None, index=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    link : instance of link function
        If no link function is provided, then the ``mmodel.family.link` is
        used.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """
    pass

def get_prediction_delta(self, exog=None, which='mean', average=False, agg_weights=None, transform=True, row_labels=None, pred_kwds=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    which : str
        The statistic that is prediction. Which statistics are available
        depends on the model.predict method.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then them mean over
        observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """
    pass

def get_prediction(self, exog=None, transform=True, which='mean', row_labels=None, average=False, agg_weights=None, pred_kwds=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    which : str
        Which statistic is to be predicted. Default is "mean".
        The available statistics and options depend on the model.
        see the model.predict docstring
    linear : bool
        Linear has been replaced by the `which` keyword and will be
        deprecated.
        If linear is True, then `which` is ignored and the linear
        prediction is returned.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then the average
        over observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    **kwargs :
        Some models can take additional keyword arguments, such as offset,
        exposure or additional exog in multi-part models like zero inflated
        models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and
        summary dataframe for the prediction.

    Notes
    -----
    Status: new in 0.14, experimental
    """
    pass

def params_transform_univariate(params, cov_params, link=None, transform=None, row_labels=None):
    """
    results for univariate, nonlinear, monotonicaly transformed parameters

    This provides transformed values, standard errors and confidence interval
    for transformations of parameters, for example in calculating rates with
    `exp(params)` in the case of Poisson or other models with exponential
    mean function.
    """
    pass