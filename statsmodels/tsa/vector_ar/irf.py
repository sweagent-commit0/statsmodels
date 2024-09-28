"""
Impulse reponse-related code
"""
import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
mat = np.array

class BaseIRAnalysis:
    """
    Base class for plotting and computing IRF-related statistics, want to be
    able to handle known and estimated processes
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        self.model = model
        self.periods = periods
        self.neqs, self.lags, self.T = (model.neqs, model.k_ar, model.nobs)
        self.order = order
        if P is None:
            sigma = model.sigma_u
            P = la.cholesky(sigma)
        self.P = P
        self.svar = svar
        self.irfs = model.ma_rep(periods)
        if svar:
            self.svar_irfs = model.svar_ma_rep(periods, P=P)
        else:
            self.orth_irfs = model.orth_ma_rep(periods, P=P)
        self.cum_effects = self.irfs.cumsum(axis=0)
        if svar:
            self.svar_cum_effects = self.svar_irfs.cumsum(axis=0)
        else:
            self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)
        if not vecm:
            self.lr_effects = model.long_run_effects()
            if svar:
                self.svar_lr_effects = np.dot(model.long_run_effects(), P)
            else:
                self.orth_lr_effects = np.dot(model.long_run_effects(), P)
        if vecm:
            self._A = util.comp_matrix(model.var_rep)
        else:
            self._A = util.comp_matrix(model.coefs)

    def plot(self, orth=False, *, impulse=None, response=None, signif=0.05, plot_params=None, figsize=(10, 10), subplot_params=None, plot_stderr=True, stderr_type='asym', repl=1000, seed=None, component=None):
        """
        Plot impulse responses

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impulse : {str, int}
            variable providing the impulse
        response : {str, int}
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are too big,
            pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict

        figsize : (float, float), default (10, 10)
            Figure size (width, height in inches)
        plot_stderr : bool, default True
            Plot standard impulse response error bands
        stderr_type : str
            'asym': default, computes asymptotic standard errors
            'mc': monte carlo standard errors (use rpl)
        repl : int, default 1000
            Number of replications for Monte Carlo and Sims-Zha standard errors
        seed : int
            np.random.seed for Monte Carlo replications
        component: array or vector of principal component indices
        """
        pass

    def plot_cum_effects(self, orth=False, *, impulse=None, response=None, signif=0.05, plot_params=None, figsize=(10, 10), subplot_params=None, plot_stderr=True, stderr_type='asym', repl=1000, seed=None):
        """
        Plot cumulative impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impulse : {str, int}
            variable providing the impulse
        response : {str, int}
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are too big,
            pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict

        figsize: (float, float), default (10, 10)
            Figure size (width, height in inches)
        plot_stderr : bool, default True
            Plot standard impulse response error bands
        stderr_type : str
            'asym': default, computes asymptotic standard errors
            'mc': monte carlo standard errors (use rpl)
        repl : int, default 1000
            Number of replications for monte carlo standard errors
        seed : int
            np.random.seed for Monte Carlo replications
        """
        pass

class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lütkepohl (2005) notation
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        BaseIRAnalysis.__init__(self, model, P=P, periods=periods, order=order, svar=svar, vecm=vecm)
        if vecm:
            self.cov_a = model.cov_var_repr
        else:
            self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma
        self._g_memo = {}

    def cov(self, orth=False):
        """
        Compute asymptotic standard errors for impulse response coefficients

        Notes
        -----
        Lütkepohl eq 3.7.5

        Returns
        -------
        """
        pass

    def errband_mc(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100):
        """
        IRF Monte Carlo integrated error bands
        """
        pass

    def err_band_sz1(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 1. Assumes symmetric error bands around
        mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        repl : int, default 1000
            Number of MC replications
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random seed
        burn : int, default 100
            Number of initial simulated obs to discard
        component : neqs x neqs array, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        pass

    def err_band_sz2(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 2.

        This method Does not assume symmetric error bands around mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        repl : int, default 1000
            Number of MC replications
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random seed
        burn : int, default 100
            Number of initial simulated obs to discard
        component : neqs x neqs array, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        pass

    def err_band_sz3(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
        """
        IRF Sims-Zha error band method 3. Does not assume symmetric error bands around mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        repl : int, default 1000
            Number of MC replications
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random seed
        burn : int, default 100
            Number of initial simulated obs to discard
        component : vector length neqs, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        pass

    def _eigval_decomp_SZ(self, irf_resim):
        """
        Returns
        -------
        W: array of eigenvectors
        eigva: list of eigenvalues
        k: matrix indicating column # of largest eigenvalue for each c_i,j
        """
        pass

    def cum_effect_cov(self, orth=False):
        """
        Compute asymptotic standard errors for cumulative impulse response
        coefficients

        Parameters
        ----------
        orth : bool

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)

        Returns
        -------
        """
        pass

    def cum_errband_mc(self, orth=False, repl=1000, signif=0.05, seed=None, burn=100):
        """
        IRF Monte Carlo integrated error bands of cumulative effect
        """
        pass

    def lr_effect_cov(self, orth=False):
        """
        Returns
        -------
        """
        pass