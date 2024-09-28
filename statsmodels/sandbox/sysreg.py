from statsmodels.regression.linear_model import GLS
import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from scipy import sparse
__all__ = ['SUR', 'Sem2SLS']

class SUR:
    """
    Seemingly Unrelated Regression

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    sigma : array_like
        M x M array where sigma[i,j] is the covariance between equation i and j
    dfk : None, 'dfk1', or 'dfk2'
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.

    Attributes
    ----------
    cholsigmainv : ndarray
        The transpose of the Cholesky decomposition of `pinv_wexog`
    df_model : ndarray
        Model degrees of freedom of each equation. p_{m} - 1 where p is
        the number of regressors for each equation m and one is subtracted
        for the constant.
    df_resid : ndarray
        Residual degrees of freedom of each equation. Number of observations
        less the number of parameters.
    endog : ndarray
        The LHS variables for each equation in the system.
        It is a M x nobs array where M is the number of equations.
    exog : ndarray
        The RHS variable for each equation in the system.
        It is a nobs x sum(p_{m}) array.  Which is just each
        RHS array stacked next to each other in columns.
    history : dict
        Contains the history of fitting the model. Probably not of interest
        if the model is fit with `igls` = False.
    iterations : int
        The number of iterations until convergence if the model is fit
        iteratively.
    nobs : float
        The number of observations of the equations.
    normalized_cov_params : ndarray
        sum(p_{m}) x sum(p_{m}) array
        :math:`\\left[X^{T}\\left(\\Sigma^{-1}\\otimes\\boldsymbol{I}\\right)X\\right]^{-1}`
    pinv_wexog : ndarray
        The pseudo-inverse of the `wexog`
    sigma : ndarray
        M x M covariance matrix of the cross-equation disturbances. See notes.
    sp_exog : CSR sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        exog1 ... exogM are on the diagonal.
    wendog : ndarray
        M * nobs x 1 array of the endogenous variables whitened by
        `cholsigmainv` and stacked into a single column.
    wexog : ndarray
        M*nobs x sum(p_{m}) array of the whitened exogenous variables.

    Notes
    -----
    All individual equations are assumed to be well-behaved, homoskedastic
    iid errors.  This is basically an extension of GLS, using sparse matrices.

    .. math:: \\Sigma=\\left[\\begin{array}{cccc}
              \\sigma_{11} & \\sigma_{12} & \\cdots & \\sigma_{1M}\\\\
              \\sigma_{21} & \\sigma_{22} & \\cdots & \\sigma_{2M}\\\\
              \\vdots & \\vdots & \\ddots & \\vdots\\\\
              \\sigma_{M1} & \\sigma_{M2} & \\cdots & \\sigma_{MM}\\end{array}\\right]

    References
    ----------
    Zellner (1962), Greene (2003)
    """

    def __init__(self, sys, sigma=None, dfk=None):
        if len(sys) % 2 != 0:
            raise ValueError('sys must be a list of pairs of endogenous and exogenous variables.  Got length %s' % len(sys))
        if dfk:
            if not dfk.lower() in ['dfk1', 'dfk2']:
                raise ValueError('dfk option %s not understood' % dfk)
        self._dfk = dfk
        M = len(sys[1::2])
        self._M = M
        exog = np.column_stack((np.asarray(sys[1::2][i]) for i in range(M)))
        self.exog = exog
        endog = np.asarray(sys[::2])
        self.endog = endog
        self.nobs = float(self.endog[0].shape[0])
        df_resid = []
        df_model = []
        [df_resid.append(self.nobs - np.linalg.matrix_rank(_)) for _ in sys[1::2]]
        [df_model.append(np.linalg.matrix_rank(_) - 1) for _ in sys[1::2]]
        self.df_resid = np.asarray(df_resid)
        self.df_model = np.asarray(df_model)
        sp_exog = sparse.lil_matrix((int(self.nobs * M), int(np.sum(self.df_model + 1))))
        self._cols = np.cumsum(np.hstack((0, self.df_model + 1)))
        for i in range(M):
            sp_exog[i * self.nobs:(i + 1) * self.nobs, self._cols[i]:self._cols[i + 1]] = sys[1::2][i]
        self.sp_exog = sp_exog.tocsr()
        if np.any(sigma):
            sigma = np.asarray(sigma)
        elif sigma is None:
            resids = []
            for i in range(M):
                resids.append(GLS(endog[i], exog[:, self._cols[i]:self._cols[i + 1]]).fit().resid)
            resids = np.asarray(resids).reshape(M, -1)
            sigma = self._compute_sigma(resids)
        self.sigma = sigma
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.initialize()

    def _compute_sigma(self, resids):
        """
        Computes the sigma matrix and update the cholesky decomposition.
        """
        pass

    def whiten(self, X):
        """
        SUR whiten method.

        Parameters
        ----------
        X : list of arrays
            Data to be whitened.

        Returns
        -------
        If X is the exogenous RHS of the system.
        ``np.dot(np.kron(cholsigmainv,np.eye(M)),np.diag(X))``

        If X is the endogenous LHS of the system.
        """
        pass

    def fit(self, igls=False, tol=1e-05, maxiter=100):
        """
        igls : bool
            Iterate until estimates converge if sigma is None instead of
            two-step GLS, which is the default is sigma is None.

        tol : float

        maxiter : int

        Notes
        -----
        This ia naive implementation that does not exploit the block
        diagonal structure. It should work for ill-conditioned `sigma`
        but this is untested.
        """
        pass

class Sem2SLS:
    """
    Two-Stage Least Squares for Simultaneous equations

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    indep_endog : dict
        A dictionary mapping the equation to the column numbers of the
        the independent endogenous regressors in each equation.
        It is assumed that the system is entered as broken up into
        LHS and RHS. For now, the values of the dict have to be sequences.
        Note that the keys for the equations should be zero-indexed.
    instruments : ndarray
        Array of the exogenous independent variables.

    Notes
    -----
    This is unfinished, and the design should be refactored.
    Estimation is done by brute force and there is no exploitation of
    the structure of the system.
    """

    def __init__(self, sys, indep_endog=None, instruments=None):
        if len(sys) % 2 != 0:
            raise ValueError('sys must be a list of pairs of endogenous and exogenous variables.  Got length %s' % len(sys))
        M = len(sys[1::2])
        self._M = M
        self.endog = sys[::2]
        self.exog = sys[1::2]
        self._K = [np.linalg.matrix_rank(_) for _ in sys[1::2]]
        self.instruments = instruments
        instr_endog = {}
        [instr_endog.setdefault(_, []) for _ in indep_endog.keys()]
        for eq_key in indep_endog:
            for varcol in indep_endog[eq_key]:
                instr_endog[eq_key].append(self.exog[eq_key][:, varcol])
        self._indep_endog = indep_endog
        _col_map = np.cumsum(np.hstack((0, self._K)))
        for eq_key in indep_endog:
            try:
                iter(indep_endog[eq_key])
            except:
                raise TypeError('The values of the indep_exog dict must be iterable. Got type %s for converter %s' % (type(indep_endog[eq_key]), eq_key))
        self.wexog = self.whiten(instr_endog)

    def whiten(self, Y):
        """
        Runs the first stage of the 2SLS.

        Returns the RHS variables that include the instruments.
        """
        pass

    def fit(self):
        """
        """
        pass

class SysResults(LikelihoodModelResults):
    """
    Not implemented yet.
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        super(SysResults, self).__init__(model, params, normalized_cov_params, scale)
        self._get_results()