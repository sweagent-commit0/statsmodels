"""
This module implements maximum likelihood-based estimation (MLE) of
Gaussian regression models for finite-dimensional observations made on
infinite-dimensional processes.

The ProcessMLE class supports regression analyses on grouped data,
where the observations within a group are dependent (they are made on
the same underlying process).  One use-case is repeated measures
regression for temporal (longitudinal) data, in which the repeated
measures occur at arbitrary real-valued time points.

The mean structure is specified as a linear model.  The covariance
parameters depend on covariates via a link function.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings

class ProcessCovariance:
    """
    A covariance model for a process indexed by a real parameter.

    An implementation of this class is based on a positive definite
    correlation function h that maps real numbers to the interval [0,
    1], such as the Gaussian (squared exponential) correlation
    function :math:`\\exp(-x^2)`.  It also depends on a positive
    scaling function `s` and a positive smoothness function `u`.
    """

    def get_cov(self, time, sc, sm):
        """
        Returns the covariance matrix for given time values.

        Parameters
        ----------
        time : array_like
            The time points for the observations.  If len(time) = p,
            a pxp covariance matrix is returned.
        sc : array_like
            The scaling parameters for the observations.
        sm : array_like
            The smoothness parameters for the observation.  See class
            docstring for details.
        """
        pass

    def jac(self, time, sc, sm):
        """
        The Jacobian of the covariance with respect to the parameters.

        See get_cov for parameters.

        Returns
        -------
        jsc : list-like
            jsc[i] is the derivative of the covariance matrix
            with respect to the i^th scaling parameter.
        jsm : list-like
            jsm[i] is the derivative of the covariance matrix
            with respect to the i^th smoothness parameter.
        """
        pass

class GaussianCovariance(ProcessCovariance):
    """
    An implementation of ProcessCovariance using the Gaussian kernel.

    This class represents a parametric covariance model for a Gaussian
    process as described in the work of Paciorek et al. cited below.

    Following Paciorek et al [1]_, the covariance between observations with
    index `i` and `j` is given by:

    .. math::

      s[i] \\cdot s[j] \\cdot h(|time[i] - time[j]| / \\sqrt{(u[i] + u[j]) /
      2}) \\cdot \\frac{u[i]^{1/4}u[j]^{1/4}}{\\sqrt{(u[i] + u[j])/2}}

    The ProcessMLE class allows linear models with this covariance
    structure to be fit using maximum likelihood (ML). The mean and
    covariance parameters of the model are fit jointly.

    The mean, scaling, and smoothing parameters can be linked to
    covariates.  The mean parameters are linked linearly, and the
    scaling and smoothing parameters use an log link function to
    preserve positivity.

    The reference of Paciorek et al. below provides more details.
    Note that here we only implement the 1-dimensional version of
    their approach.

    References
    ----------
    .. [1] Paciorek, C. J. and Schervish, M. J. (2006). Spatial modeling using
        a new class of nonstationary covariance functions. Environmetrics,
        17:483–506.
        https://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf
    """

class ProcessMLE(base.LikelihoodModel):
    """
    Fit a Gaussian mean/variance regression model.

    This class fits a one-dimensional Gaussian process model with
    parametrized mean and covariance structures to grouped data.  For
    each group, there is an independent realization of a latent
    Gaussian process indexed by an observed real-valued time
    variable..  The data consist of the Gaussian process observed at a
    finite number of `time` values.

    The process mean and variance can be lined to covariates.  The
    mean structure is linear in the covariates.  The covariance
    structure is non-stationary, and is defined parametrically through
    'scaling', and 'smoothing' parameters.  The covariance of the
    process between two observations in the same group is a function
    of the distance between the time values of the two observations.
    The scaling and smoothing parameters can be linked to covariates.

    The observed data are modeled as the sum of the Gaussian process
    realization and (optionally) independent white noise.  The standard
    deviation of the white noise can be linked to covariates.

    The data should be provided in 'long form', with a group label to
    indicate which observations belong to the same group.
    Observations in different groups are always independent.

    Parameters
    ----------
    endog : array_like
        The dependent variable.
    exog : array_like
        The design matrix for the mean structure
    exog_scale : array_like
        The design matrix for the scaling structure
    exog_smooth : array_like
        The design matrix for the smoothness structure
    exog_noise : array_like
        The design matrix for the additive white noise. The
        linear predictor is the log of the white noise standard
        deviation.  If None, there is no additive noise (the
        process is observed directly).
    time : array_like (1-dimensional)
        The univariate index values, used to calculate distances
        between observations in the same group, which determines
        their correlations.
    groups : array_like (1-dimensional)
        The group values.
    cov : a ProcessCovariance instance
        Defaults to GaussianCovariance.
    """

    def __init__(self, endog, exog, exog_scale, exog_smooth, exog_noise, time, groups, cov=None, **kwargs):
        super(ProcessMLE, self).__init__(endog, exog, exog_scale=exog_scale, exog_smooth=exog_smooth, exog_noise=exog_noise, time=time, groups=groups, **kwargs)
        self._has_noise = exog_noise is not None
        xnames = []
        if hasattr(exog, 'columns'):
            xnames = list(exog.columns)
        else:
            xnames = ['Mean%d' % j for j in range(exog.shape[1])]
        if hasattr(exog_scale, 'columns'):
            xnames += list(exog_scale.columns)
        else:
            xnames += ['Scale%d' % j for j in range(exog_scale.shape[1])]
        if hasattr(exog_smooth, 'columns'):
            xnames += list(exog_smooth.columns)
        else:
            xnames += ['Smooth%d' % j for j in range(exog_smooth.shape[1])]
        if self._has_noise:
            if hasattr(exog_noise, 'columns'):
                xnames += list(exog_noise.columns)
            else:
                xnames += ['Noise%d' % j for j in range(exog_noise.shape[1])]
        self.data.param_names = xnames
        if cov is None:
            cov = GaussianCovariance()
        self.cov = cov
        _check_args(endog, exog, exog_scale, exog_smooth, exog_noise, time, groups)
        groups_ix = collections.defaultdict(lambda: [])
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self._groups_ix = groups_ix
        self.verbose = False
        self.k_exog = self.exog.shape[1]
        self.k_scale = self.exog_scale.shape[1]
        self.k_smooth = self.exog_smooth.shape[1]
        if self._has_noise:
            self.k_noise = self.exog_noise.shape[1]

    def unpack(self, z):
        """
        Split the packed parameter vector into blocks.
        """
        pass

    def loglike(self, params):
        """
        Calculate the log-likelihood function for the model.

        Parameters
        ----------
        params : array_like
            The packed parameters for the model.

        Returns
        -------
        The log-likelihood value at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """
        pass

    def score(self, params):
        """
        Calculate the score function for the model.

        Parameters
        ----------
        params : array_like
            The packed parameters for the model.

        Returns
        -------
        The score vector at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """
        pass

    def fit(self, start_params=None, method=None, maxiter=None, **kwargs):
        """
        Fit a grouped Gaussian process regression using MLE.

        Parameters
        ----------
        start_params : array_like
            Optional starting values.
        method : str or array of str
            Method or sequence of methods for scipy optimize.
        maxiter : int
            The maximum number of iterations in the optimization.

        Returns
        -------
        An instance of ProcessMLEResults.
        """
        pass

    def covariance(self, time, scale_params, smooth_params, scale_data, smooth_data):
        """
        Returns a Gaussian process covariance matrix.

        Parameters
        ----------
        time : array_like
            The time points at which the fitted covariance matrix is
            calculated.
        scale_params : array_like
            The regression parameters for the scaling part
            of the covariance structure.
        smooth_params : array_like
            The regression parameters for the smoothing part
            of the covariance structure.
        scale_data : DataFrame
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth_data : DataFrame
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should contain data arrays whose
        columns align with the fitted scaling and smoothing parameters.

        The covariance is only for the Gaussian process and does not include
        the white noise variance.
        """
        pass

    def predict(self, params, exog=None, *args, **kwargs):
        """
        Obtain predictions of the mean structure.

        Parameters
        ----------
        params : array_like
            The model parameters, may be truncated to include only mean
            parameters.
        exog : array_like
            The design matrix for the mean structure.  If not provided,
            the model's design matrix is used.
        """
        pass

class ProcessMLEResults(base.GenericLikelihoodModelResults):
    """
    Results class for Gaussian process regression models.
    """

    def __init__(self, model, mlefit):
        super(ProcessMLEResults, self).__init__(model, mlefit)
        pa = model.unpack(mlefit.params)
        self.mean_params = pa[0]
        self.scale_params = pa[1]
        self.smooth_params = pa[2]
        self.no_params = pa[3]
        self.df_resid = model.endog.shape[0] - len(mlefit.params)
        self.k_exog = self.model.exog.shape[1]
        self.k_scale = self.model.exog_scale.shape[1]
        self.k_smooth = self.model.exog_smooth.shape[1]
        self._has_noise = model._has_noise
        if model._has_noise:
            self.k_noise = self.model.exog_noise.shape[1]

    def covariance(self, time, scale, smooth):
        """
        Returns a fitted covariance matrix.

        Parameters
        ----------
        time : array_like
            The time points at which the fitted covariance
            matrix is calculated.
        scale : array_like
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth : array_like
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should be data arrays whose
        columns align with the fitted scaling and smoothing parameters.
        """
        pass