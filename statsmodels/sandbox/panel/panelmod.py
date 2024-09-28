"""
Sandbox Panel Estimators

References
-----------

Baltagi, Badi H. `Econometric Analysis of Panel Data.` 4th ed. Wiley, 2008.
"""
from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
__all__ = ['PanelModel']
from pandas import Panel

def group(X):
    """
    Returns unique numeric values for groups without sorting.

    Examples
    --------
    >>> X = np.array(['a','a','b','c','b','c'])
    >>> group(X)
    >>> g
    array([ 0.,  0.,  1.,  2.,  1.,  2.])
    """
    pass

def repanel_cov(groups, sigmas):
    """calculate error covariance matrix for random effects model

    Parameters
    ----------
    groups : ndarray, (nobs, nre) or (nobs,)
        array of group/category observations
    sigma : ndarray, (nre+1,)
        array of standard deviations of random effects,
        last element is the standard deviation of the
        idiosyncratic error

    Returns
    -------
    omega : ndarray, (nobs, nobs)
        covariance matrix of error
    omegainv : ndarray, (nobs, nobs)
        inverse covariance matrix of error
    omegainvsqrt : ndarray, (nobs, nobs)
        squareroot inverse covariance matrix of error
        such that omega = omegainvsqrt * omegainvsqrt.T

    Notes
    -----
    This does not use sparse matrices and constructs nobs by nobs
    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero
    """
    pass

class PanelData(Panel):
    pass

class PanelModel:
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ----------
    endog : array_like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.Panel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """

    def __init__(self, endog=None, exog=None, panel=None, time=None, xtnames=None, equation=None, panel_data=None):
        if panel_data is None:
            self.initialize(endog, exog, panel, time, xtnames, equation)

    def initialize(self, endog, exog, panel, time, xtnames, equation):
        """
        Initialize plain array model.

        See PanelModel
        """
        pass

    def _group_mean(self, X, index='oneway', counts=False, dummies=False):
        """
        Get group means of X by time or by panel.

        index default is panel
        """
        pass

    def fit(self, model=None, method=None, effects='oneway'):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE, optional
        model :
                between
                fixed
                random
                pooled
                [gmm]
        effects :
                oneway
                time
                twoway
        femethod : demeaned (only one implemented)
                   WLS
        remethod :
                swar -
                amemiya
                nerlove
                walhus


        Notes
        -----
        This is unfinished.  None of the method arguments work yet.
        Only oneway effects should work.
        """
        pass

class SURPanel(PanelModel):
    pass

class SEMPanel(PanelModel):
    pass

class DynamicPanel(PanelModel):
    pass
if __name__ == '__main__':
    import numpy.lib.recfunctions as nprf
    import pandas
    from pandas import Panel
    import statsmodels.api as sm
    data = sm.datasets.grunfeld.load()
    endog = data.endog[:-20]
    fullexog = data.exog[:-20]
    panel_arr = nprf.append_fields(fullexog, 'investment', endog, float, usemask=False)
    panel_df = pandas.DataFrame(panel_arr)
    panel_panda = panel_df.set_index(['year', 'firm']).to_panel()
    exog = fullexog[['value', 'capital']].view(float).reshape(-1, 2)
    exog = sm.add_constant(exog, prepend=False)
    panel = group(fullexog['firm'])
    year = fullexog['year']
    panel_mod = PanelModel(endog, exog, panel, year, xtnames=['firm', 'year'], equation='invest value capital')
    panel_ols = panel_mod.fit(model='pooled')
    panel_be = panel_mod.fit(model='between', effects='oneway')
    panel_fe = panel_mod.fit(model='fixed', effects='oneway')
    panel_bet = panel_mod.fit(model='between', effects='time')
    panel_fet = panel_mod.fit(model='fixed', effects='time')
    panel_fe2 = panel_mod.fit(model='fixed', effects='twoways')
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    nobs = groups.shape[0]
    groupuniq = np.unique(groups)
    periods = np.array([0, 1, 2, 1, 2, 0, 1, 2])
    perioduniq = np.unique(periods)
    dummygr = (groups[:, None] == groupuniq).astype(float)
    dummype = (periods[:, None] == perioduniq).astype(float)
    sigma = 1.0
    sigmagr = np.sqrt(2.0)
    sigmape = np.sqrt(3.0)
    dummyall = np.c_[sigmagr * dummygr, sigmape * dummype]
    omega = np.dot(dummyall, dummyall.T) + sigma * np.eye(nobs)
    print(omega)
    print(np.linalg.cholesky(omega))
    ev, evec = np.linalg.eigh(omega)
    omegainv = np.dot(evec, (1 / ev * evec).T)
    omegainv2 = np.linalg.inv(omega)
    omegacomp = np.dot(evec, (ev * evec).T)
    print(np.max(np.abs(omegacomp - omega)))
    print(np.max(np.abs(np.dot(omegainv, omega) - np.eye(nobs))))
    omegainvhalf = evec / np.sqrt(ev)
    print(np.max(np.abs(np.dot(omegainvhalf, omegainvhalf.T) - omegainv)))
    sigmas2 = np.array([sigmagr, sigmape, sigma])
    groups2 = np.column_stack((groups, periods))
    omega_, omegainv_, omegainvhalf_ = repanel_cov(groups2, sigmas2)
    print(np.max(np.abs(omega_ - omega)))
    print(np.max(np.abs(omegainv_ - omegainv)))
    print(np.max(np.abs(omegainvhalf_ - omegainvhalf)))
    Pgr = reduce(np.dot, [dummygr, np.linalg.inv(np.dot(dummygr.T, dummygr)), dummygr.T])
    Qgr = np.eye(nobs) - Pgr
    print(np.max(np.abs(np.dot(Qgr, groups))))