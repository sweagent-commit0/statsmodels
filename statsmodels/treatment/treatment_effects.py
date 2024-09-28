"""Treatment effect estimators

follows largely Stata's teffects in Stata 13 manual

Created on Tue Jun  9 22:45:23 2015

Author: Josef Perktold
License: BSD-3

currently available

                     ATE        POM_0        POM_1
res_ipw       230.688598  3172.774059  3403.462658
res_aipw     -230.989201  3403.355253  3172.366052
res_aipw_wls -227.195618  3403.250651  3176.055033
res_ra       -239.639211  3403.242272  3163.603060
res_ipwra    -229.967078  3403.335639  3173.368561


Lots of todos, just the beginning, but most effects are available but not
standard errors, and no code structure that has a useful pattern

see https://github.com/statsmodels/statsmodels/issues/2443

Note: script requires cattaneo2 data file from Stata 14, hardcoded file path
could be loaded with webuse

"""
import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent

def _mom_ate(params, endog, tind, prob, weighted=True):
    """moment condition for average treatment effect

    This does not include a moment condition for potential outcome mean (POM).

    """
    pass

def _mom_atm(params, endog, tind, prob, weighted=True):
    """moment conditions for average treatment means (POM)

    moment conditions are POM0 and POM1
    """
    pass

def _mom_ols(params, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    moment conditions are POM0 and POM1

    """
    pass

def _mom_ols_te(tm, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    first moment is ATE
    second moment is POM0  (control)

    """
    pass

def ate_ipw(endog, tind, prob, weighted=True, probt=None):
    """average treatment effect based on basic inverse propensity weighting.

    """
    pass

class _TEGMMGeneric1(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome, exclude_tmoms=False, **kwargs):
        super(_TEGMMGeneric1, self).__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome
        self.exclude_tmoms = exclude_tmoms
        self.__dict__.update(kwargs)
        if self.data.xnames is None:
            self.data.xnames = []
        if exclude_tmoms:
            self.k_select = 0
        else:
            self.k_select = len(res_select.model.data.param_names)
        if exclude_tmoms:
            self.prob = self.results_select.predict()
        else:
            self.prob = None

class _TEGMM(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome):
        super(_TEGMM, self).__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome
        if self.data.xnames is None:
            self.data.xnames = []

class _IPWGMM(_TEGMMGeneric1):
    """ GMM for aipw treatment effect and potential outcome

    uses unweighted outcome regression
    """

class _AIPWGMM(_TEGMMGeneric1):
    """ GMM for aipw treatment effect and potential outcome

    uses unweighted outcome regression
    """

class _AIPWWLSGMM(_TEGMMGeneric1):
    """ GMM for aipw-wls treatment effect and potential outcome

    uses weighted outcome regression
    """

class _RAGMM(_TEGMMGeneric1):
    """GMM for regression adjustment treatment effect and potential outcome

    uses unweighted outcome regression
    """

class _IPWRAGMM(_TEGMMGeneric1):
    """ GMM for ipwra treatment effect and potential outcome
    """

class TreatmentEffectResults(ContrastResults):
    """
    Results class for treatment effect estimation

    Parameters
    ----------
    teff : instance of TreatmentEffect class
    results_gmm : instance of GMMResults class
    method : string
        Method and estimator of treatment effect.
    kwds: dict
        Other keywords with additional information.

    Notes
    -----
    This class is a subclass of ContrastResults and inherits methods like
    summary, summary_frame and conf_int. Attributes correspond to a z-test
    given by ``GMMResults.t_test``.
    """

    def __init__(self, teff, results_gmm, method, **kwds):
        super().__init__()
        k_params = len(results_gmm.params)
        constraints = np.zeros((3, k_params))
        constraints[0, 0] = 1
        constraints[1, 1] = 1
        constraints[2, :2] = [1, 1]
        tt = results_gmm.t_test(constraints)
        self.__dict__.update(tt.__dict__)
        self.teff = teff
        self.results_gmm = results_gmm
        self.method = method
        self.__dict__.update(kwds)
        self.c_names = ['ATE', 'POM0', 'POM1']
doc_params_returns = 'Parameters\n----------\nreturn_results : bool\n    If True, then a results instance is returned.\n    If False, just ATE, POM0 and POM1 are returned.\neffect_group : {"all", 0, 1}\n    ``effectgroup`` determines for which population the effects are\n    estimated.\n    If effect_group is "all", then sample average treatment effect and\n    potential outcomes are returned\n    If effect_group is 1 or "treated", then effects on treated are\n    returned.\n    If effect_group is 0, "treated" or "control", then effects on\n    untreated, i.e. control group, are returned.\ndisp : bool\n    Indicates whether the scipy optimizer should display the\n    optimization results\n\nReturns\n-------\nTreatmentEffectsResults instance or tuple (ATE, POM0, POM1)\n'
doc_params_returns2 = 'Parameters\n----------\nreturn_results : bool\n    If True, then a results instance is returned.\n    If False, just ATE, POM0 and POM1 are returned.\ndisp : bool\n    Indicates whether the scipy optimizer should display the\n    optimization results\n\nReturns\n-------\nTreatmentEffectsResults instance or tuple (ATE, POM0, POM1)\n'

class TreatmentEffect(object):
    """
    Estimate average treatment effect under conditional independence

    .. versionadded:: 0.14.0

    This class estimates treatment effect and potential outcome using 5
    different methods, ipw, ra, aipw, aipw-wls, ipw-ra.
    Standard errors and inference are based on the joint GMM representation of
    selection or treatment model, outcome model and effect functions.

    Parameters
    ----------
    model : instance of a model class
        The model class should contain endog and exog for the outcome model.
    treatment : ndarray
        indicator array for observations with treatment (1) or without (0)
    results_select : results instance
        The results instance for the treatment or selection model.
    _cov_type : "HC0"
        Internal keyword. The keyword oes not affect GMMResults which always
        corresponds to HC0 standard errors.
    kwds : keyword arguments
        currently not used

    Notes
    -----
    The outcome model is currently limited to a linear model based on OLS.
    Other outcome models, like Logit and Poisson, will become available in
    future.

    See `Treatment Effect notebook
    <../examples/notebooks/generated/treatment_effect.html>`__
    for an overview.

    """

    def __init__(self, model, treatment, results_select=None, _cov_type='HC0', **kwds):
        self.__dict__.update(kwds)
        self.treatment = np.asarray(treatment)
        self.treat_mask = treat_mask = treatment == 1
        if results_select is not None:
            self.results_select = results_select
            self.prob_select = results_select.predict()
        self.model_pool = model
        endog = model.endog
        exog = model.exog
        self.nobs = endog.shape[0]
        self._cov_type = _cov_type
        mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
        self.results0 = mod0.fit(cov_type=_cov_type)
        mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
        self.results1 = mod1.fit(cov_type=_cov_type)
        self.exog_grouped = np.concatenate((mod0.exog, mod1.exog), axis=0)
        self.endog_grouped = np.concatenate((mod0.endog, mod1.endog), axis=0)

    @classmethod
    def from_data(cls, endog, exog, treatment, model='ols', **kwds):
        """create models from data

        not yet implemented

        """
        pass

    def ipw(self, return_results=True, effect_group='all', disp=False):
        """Inverse Probability Weighted treatment effect estimation.

        Parameters
        ----------
        return_results : bool
            If True, then a results instance is returned.
            If False, just ATE, POM0 and POM1 are returned.
        effect_group : {"all", 0, 1}
            ``effectgroup`` determines for which population the effects are
            estimated.
            If effect_group is "all", then sample average treatment effect and
            potential outcomes are returned.
            If effect_group is 1 or "treated", then effects on treated are
            returned.
            If effect_group is 0, "treated" or "control", then effects on
            untreated, i.e. control group, are returned.
        disp : bool
            Indicates whether the scipy optimizer should display the
            optimization results

        Returns
        -------
        TreatmentEffectsResults instance or tuple (ATE, POM0, POM1)

        See Also
        --------
        TreatmentEffectsResults
        """
        pass

    @Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
    def ra(self, return_results=True, effect_group='all', disp=False):
        """
        Regression Adjustment treatment effect estimation.
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults
        """
        pass

    @Substitution(params_returns=indent(doc_params_returns2, ' ' * 8))
    def aipw(self, return_results=True, disp=False):
        """
        ATE and POM from double robust augmented inverse probability weighting
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
        pass

    @Substitution(params_returns=indent(doc_params_returns2, ' ' * 8))
    def aipw_wls(self, return_results=True, disp=False):
        """
        ATE and POM from double robust augmented inverse probability weighting.

        This uses weighted outcome regression, while `aipw` uses unweighted
        outcome regression.
        Option for effect on treated or on untreated is not available.
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
        pass

    @Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
    def ipw_ra(self, return_results=True, effect_group='all', disp=False):
        """
        ATE and POM from inverse probability weighted regression adjustment.

        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
        pass