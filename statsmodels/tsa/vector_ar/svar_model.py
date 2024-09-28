"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults

class SVAR(tsbase.TimeSeriesModel):
    """
    Fit VAR and then estimate structural components of A and B, defined:

    .. math:: Ay_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + B\\var(\\epsilon_t)

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable. The independent variable.
    dates : array_like
        must match number of rows of endog
    svar_type : str
        "A" - estimate structural parameters of A matrix, B assumed = I
        "B" - estimate structural parameters of B matrix, A assumed = I
        "AB" - estimate structural parameters indicated in both A and B matrix
    A : array_like
        neqs x neqs with unknown parameters marked with 'E' for estimate
    B : array_like
        neqs x neqs with unknown parameters marked with 'E' for estimate

    References
    ----------
    Hamilton (1994) Time Series Analysis
    """
    y = deprecated_alias('y', 'endog', remove_version='0.11.0')

    def __init__(self, endog, svar_type, dates=None, freq=None, A=None, B=None, missing='none'):
        super().__init__(endog, None, dates, freq, missing=missing)
        self.neqs = self.endog.shape[1]
        types = ['A', 'B', 'AB']
        if svar_type not in types:
            raise ValueError('SVAR type not recognized, must be in ' + str(types))
        self.svar_type = svar_type
        svar_ckerr(svar_type, A, B)
        self.A_original = A
        self.B_original = B
        if A is None:
            A = np.identity(self.neqs)
            self.A_mask = A_mask = np.zeros(A.shape, dtype=bool)
        else:
            A_mask = np.logical_or(A == 'E', A == 'e')
            self.A_mask = A_mask
        if B is None:
            B = np.identity(self.neqs)
            self.B_mask = B_mask = np.zeros(B.shape, dtype=bool)
        else:
            B_mask = np.logical_or(B == 'E', B == 'e')
            self.B_mask = B_mask
        Anum = np.zeros(A.shape, dtype=float)
        Anum[~A_mask] = A[~A_mask]
        Anum[A_mask] = np.nan
        self.A = Anum
        Bnum = np.zeros(B.shape, dtype=float)
        Bnum[~B_mask] = B[~B_mask]
        Bnum[B_mask] = np.nan
        self.B = Bnum

    def fit(self, A_guess=None, B_guess=None, maxlags=None, method='ols', ic=None, trend='c', verbose=False, s_method='mle', solver='bfgs', override=False, maxiter=500, maxfun=500):
        """
        Fit the SVAR model and solve for structural parameters

        Parameters
        ----------
        A_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in A.
        B_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in B.
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend, str {"c", "ct", "ctt", "n"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - co constant, no trend
            Note that these are prepended to the columns of the dataset.
        s_method : {'mle'}
            Estimation method for structural parameters
        solver : {'nm', 'newton', 'bfgs', 'cg', 'ncg', 'powell'}
            Solution method
            See statsmodels.base for details
        override : bool, default False
            If True, returns estimates of A and B without checking
            order or rank condition
        maxiter : int, default 500
            Number of iterations to perform in solution method
        maxfun : int
            Number of function evaluations to perform

        Notes
        -----
        Lütkepohl pp. 146-153
        Hamilton pp. 324-336

        Returns
        -------
        est : SVARResults
        """
        pass

    def _get_init_params(self, A_guess, B_guess):
        """
        Returns either the given starting or .1 if none are given.
        """
        pass

    def _estimate_svar(self, start_params, lags, maxiter, maxfun, trend='c', solver='nm', override=False):
        """
        lags : int
        trend : {str, None}
            As per above
        """
        pass

    def loglike(self, params):
        """
        Loglikelihood for SVAR model

        Notes
        -----
        This method assumes that the autoregressive parameters are
        first estimated, then likelihood with structural parameters
        is estimated
        """
        pass

    def score(self, AB_mask):
        """
        Return the gradient of the loglike at AB_mask.

        Parameters
        ----------
        AB_mask : unknown values of A and B matrix concatenated

        Notes
        -----
        Return numerical gradient
        """
        pass

    def hessian(self, AB_mask):
        """
        Returns numerical hessian.
        """
        pass

    def _solve_AB(self, start_params, maxiter, override=False, solver='bfgs'):
        """
        Solves for MLE estimate of structural parameters

        Parameters
        ----------

        override : bool, default False
            If True, returns estimates of A and B without checking
            order or rank condition
        solver : str or None, optional
            Solver to be used. The default is 'nm' (Nelder-Mead). Other
            choices are 'bfgs', 'newton' (Newton-Raphson), 'cg'
            conjugate, 'ncg' (non-conjugate gradient), and 'powell'.
        maxiter : int, optional
            The maximum number of iterations. Default is 500.

        Returns
        -------
        A_solve, B_solve: ML solutions for A, B matrices
        """
        pass

class SVARProcess(VARProcess):
    """
    Class represents a known SVAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    intercept : ndarray (length k)
    sigma_u : ndarray (k x k)
    names : sequence (length k)
    A : neqs x neqs np.ndarray with unknown parameters marked with 'E'
    A_mask : neqs x neqs mask array with known parameters masked
    B : neqs x neqs np.ndarry with unknown parameters marked with 'E'
    B_mask : neqs x neqs mask array with known parameters masked
    """

    def __init__(self, coefs, intercept, sigma_u, A_solve, B_solve, names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.A_solve = A_solve
        self.B_solve = B_solve
        self.names = names

    def orth_ma_rep(self, maxn=10, P=None):
        """

        Unavailable for SVAR
        """
        pass

    def svar_ma_rep(self, maxn=10, P=None):
        """

        Compute Structural MA coefficient matrices using MLE
        of A, B
        """
        pass

class SVARResults(SVARProcess, VARResults):
    """
    Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in `endog`.
    dates

    Attributes
    ----------
    aic
    bic
    bse
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    cov_params
    dates
    detomega
    df_model : int
    df_resid : int
    endog
    endog_lagged
    fittedvalues
    fpe
    intercept
    info_criteria
    k_ar : int
    k_trend : int
    llf
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params
    k_ar : int
        Order of VAR process
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    pvalue
    names : list
        variables names
    resid
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle
    stderr
    trenorder
    tvalues
    """
    _model_type = 'SVAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order, A=None, B=None, A_mask=None, B_mask=None, model=None, trend='c', names=None, dates=None):
        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates
        self.n_totobs, self.neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        k_trend = util.get_trendorder(trend)
        if k_trend > 0:
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.k_trend = k_trend
        self.k_exog = k_trend
        self.trendorder = trendorder
        self.exog_names = util.make_lag_names(names, lag_order, k_trend)
        self.params = params
        self.sigma_u = sigma_u
        reshaped = self.params[self.k_trend:]
        reshaped = reshaped.reshape((lag_order, self.neqs, self.neqs))
        intercept = self.params[0]
        coefs = reshaped.swapaxes(1, 2).copy()
        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask
        super().__init__(coefs, intercept, sigma_u, A, B, names=names)

    def irf(self, periods=10, var_order=None):
        """
        Analyze structural impulse responses to shocks in system

        Parameters
        ----------
        periods : int

        Returns
        -------
        irf : IRAnalysis
        """
        pass

    def sirf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        pass