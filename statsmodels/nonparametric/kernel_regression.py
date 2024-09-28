"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
    Princeton University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
    and Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
    with Categorical and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)
[5] Liu, R., Yang, L. "Kernel estimation of multivariate
    cumulative distribution function."
    Journal of Nonparametric Statistics (2008)
[6] Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
    with Categorical and Continuous Data." Working Paper
[7] Li, Q., Racine, J. "Cross-validated local linear nonparametric
    regression" Statistica Sinica 14(2004), pp. 485-512
[8] Racine, J.: "Consistent Significance Testing for Nonparametric
        Regression" Journal of Business & Economics Statistics
[9] Racine, J., Hart, J., Li, Q., "Testing the Significance of
        Categorical Predictor Variables in Nonparametric Regression
        Models", 2006, Econometric Reviews 25, 523-544

"""
import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, LeaveOneOut, _get_type_pos, _adjust_shape, _compute_min_std_IQR, kernel_func
__all__ = ['KernelReg', 'KernelCensoredReg']

class KernelReg(GenericKDE):
    """
    Nonparametric kernel regression class.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``.
    Note that the "local constant" type of regression provided here is also
    known as Nadaraya-Watson kernel regression; "local linear" is an extension
    of that which suffers less from bias issues at the edge of the support. Note
    that specifying a custom kernel works only with "local linear" kernel
    regression. For example, a custom ``tricube`` kernel yields LOESS regression.

    Parameters
    ----------
    endog : array_like
        This is the dependent variable.
    exog : array_like
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    var_type : str
        The type of the variables, one character per variable:

            - c: continuous
            - u: unordered (discrete)
            - o: ordered (discrete)

    reg_type : {'lc', 'll'}, optional
        Type of regression estimator. 'lc' means local constant and
        'll' local Linear estimator.  Default is 'll'
    bw : str or array_like, optional
        Either a user-specified bandwidth or the method for bandwidth
        selection. If a string, valid values are 'cv_ls' (least-squares
        cross-validation) and 'aic' (AIC Hurvich bandwidth estimation).
        Default is 'cv_ls'. User specified bandwidth must have as many
        entries as the number of variables.
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation.

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters.
    """

    def __init__(self, endog, exog, var_type, reg_type='ll', bw='cv_ls', ckertype='gaussian', okertype='wangryzin', ukertype='aitchisonaitken', defaults=None):
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.ckertype = ckertype
        self.okertype = okertype
        self.ukertype = ukertype
        if not (self.ckertype in kernel_func and self.ukertype in kernel_func and (self.okertype in kernel_func)):
            raise ValueError('user specified kernel must be a supported kernel from statsmodels.nonparametric.kernels.')
        self.k_vars = len(self.var_type)
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.k_vars)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not isinstance(bw, str):
            bw = np.asarray(bw)
            if len(bw) != self.k_vars:
                raise ValueError('bw must have the same dimension as the number of variables.')
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def _est_loc_linear(self, bw, endog, exog, data_predict):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth value(s).
        endog : 1D array_like
            The dependent variable.
        exog : 1D or 2D array_like
            The independent variable(s).
        data_predict : 1D array_like of length K, where K is the number of variables.
            The point at which the density is estimated.

        Returns
        -------
        D_x : array_like
            The value of the conditional mean at `data_predict`.

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas.
        Unlike other methods, this one requires that `data_predict` be 1D.
        """
        pass

    def _est_loc_constant(self, bw, endog, exog, data_predict):
        """
        Local constant estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw : array_like
            Array of bandwidth value(s).
        endog : 1D array_like
            The dependent variable.
        exog : 1D or 2D array_like
            The independent variable(s).
        data_predict : 1D or 2D array_like
            The point(s) at which the density is estimated.

        Returns
        -------
        G : ndarray
            The value of the conditional mean at `data_predict`.
        B_x : ndarray
            The marginal effects.
        """
        pass

    def aic_hurvich(self, bw, func=None):
        """
        Computes the AIC Hurvich criteria for the estimation of the bandwidth.

        Parameters
        ----------
        bw : str or array_like
            See the ``bw`` parameter of `KernelReg` for details.

        Returns
        -------
        aic : ndarray
            The AIC Hurvich criteria, one element for each variable.
        func : None
            Unused here, needed in signature because it's used in `cv_loo`.

        References
        ----------
        See ch.2 in [1] and p.35 in [2].
        """
        pass

    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out estimator.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth values.
        func : callable function
            Returns the estimator of g(x).  Can be either ``_est_loc_constant``
            (local constant) or ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L : float
            The value of the CV function.

        Notes
        -----
        Calculates the cross-validation least-squares function. This function
        is minimized by compute_bw to calculate the optimal value of `bw`.

        For details see p.35 in [2]

        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths
        """
        pass

    def r_squared(self):
        """
        Returns the R-Squared for the nonparametric regression.

        Notes
        -----
        For more details see p.45 in [2]
        The R-Squared is calculated by:

        .. math:: R^{2}=\\frac{\\left[\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})(\\hat{Y_{i}}-\\bar{y}\\right]^{2}}{\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})^{2}\\sum_{i=1}^{n}(\\hat{Y_{i}}-\\bar{y})^{2}},

        where :math:`\\hat{Y_{i}}` is the mean calculated in `fit` at the exog
        points.
        """
        pass

    def fit(self, data_predict=None):
        """
        Returns the mean and marginal effects at the `data_predict` points.

        Parameters
        ----------
        data_predict : array_like, optional
            Points at which to return the mean and marginal effects.  If not
            given, ``data_predict == exog``.

        Returns
        -------
        mean : ndarray
            The regression result for the mean (i.e. the actual curve).
        mfx : ndarray
            The marginal effects, i.e. the partial derivatives of the mean.
        """
        pass

    def sig_test(self, var_pos, nboot=50, nested_res=25, pivot=False):
        """
        Significance test for the variables in the regression.

        Parameters
        ----------
        var_pos : sequence
            The position of the variable in exog to be tested.

        Returns
        -------
        sig : str
            The level of significance:

                - `*` : at 90% confidence level
                - `**` : at 95% confidence level
                - `***` : at 99* confidence level
                - "Not Significant" : if not significant
        """
        pass

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KernelReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   N = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        pass

    def _compute_dispersion(self, data):
        """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
        pass

class KernelCensoredReg(KernelReg):
    """
    Nonparametric censored regression.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``,
    where y is left-censored.  Left censored variable Y is defined as
    ``Y = min {Y', L}`` where ``L`` is the value at which ``Y`` is censored
    and ``Y'`` is the true value of the variable.

    Parameters
    ----------
    endog : list with one element which is array_like
        This is the dependent variable.
    exog : list
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    dep_type : str
        The type of the dependent variable(s)
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    reg_type : str
        Type of regression estimator
        lc: Local Constant Estimator
        ll: Local Linear Estimator
    bw : array_like
        Either a user-specified bandwidth or
        the method for bandwidth selection.
        cv_ls: cross-validation least squares
        aic: AIC Hurvich Estimator
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    censor_val : float
        Value at which the dependent variable is censored
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters
    """

    def __init__(self, endog, exog, var_type, reg_type, bw='cv_ls', ckertype='gaussian', ukertype='aitchison_aitken_reg', okertype='wangryzin_reg', censor_val=0, defaults=None):
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.ckertype = ckertype
        self.okertype = okertype
        self.ukertype = ukertype
        if not (self.ckertype in kernel_func and self.ukertype in kernel_func and (self.okertype in kernel_func)):
            raise ValueError('user specified kernel must be a supported kernel from statsmodels.nonparametric.kernels.')
        self.k_vars = len(self.var_type)
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.k_vars)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        self.censor_val = censor_val
        if self.censor_val is not None:
            self.censored(censor_val)
        else:
            self.W_in = np.ones((self.nobs, 1))
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KernelCensoredReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _est_loc_linear(self, bw, endog, exog, data_predict, W):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth value(s)
        endog : 1D array_like
            The dependent variable
        exog : 1D or 2D array_like
            The independent variable(s)
        data_predict : 1D array_like of length K, where K is
            the number of variables. The point at which
            the density is estimated

        Returns
        -------
        D_x : array_like
            The value of the conditional mean at data_predict

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas
        Unlike other methods, this one requires that data_predict be 1D
        """
        pass

    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out
        estimator

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth values
        func : callable function
            Returns the estimator of g(x).
            Can be either ``_est_loc_constant`` (local constant) or
            ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L : float
            The value of the CV function

        Notes
        -----
        Calculates the cross-validation least-squares
        function. This function is minimized by compute_bw
        to calculate the optimal value of bw

        For details see p.35 in [2]

        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths
        """
        pass

    def fit(self, data_predict=None):
        """
        Returns the marginal effects at the data_predict points.
        """
        pass

class TestRegCoefC:
    """
    Significance test for continuous variables in a nonparametric regression.

    The null hypothesis is ``dE(Y|X)/dX_not_i = 0``, the alternative hypothesis
    is ``dE(Y|X)/dX_not_i != 0``.

    Parameters
    ----------
    model : KernelReg instance
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars : tuple, list of integers, array_like
        index of position of the continuous variables to be tested
        for significance. E.g. (1,3,5) jointly tests variables at
        position 1,3 and 5 for significance.
    nboot : int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400
    nested_res : int
        Number of nested resamples used to calculate lambda.
        Must enable the pivot option
    pivot : bool
        Pivot the test statistic by dividing by its standard error
        Significantly increases computational time. But pivot statistics
        have more desirable properties
        (See references)

    Attributes
    ----------
    sig : str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class allows testing of joint hypothesis as long as all variables
    are continuous.

    References
    ----------
    Racine, J.: "Consistent Significance Testing for Nonparametric Regression"
    Journal of Business & Economics Statistics.

    Chapter 12 in [1].
    """

    def __init__(self, model, test_vars, nboot=400, nested_res=400, pivot=False):
        self.nboot = nboot
        self.nres = nested_res
        self.test_vars = test_vars
        self.model = model
        self.bw = model.bw
        self.var_type = model.var_type
        self.k_vars = len(self.var_type)
        self.endog = model.endog
        self.exog = model.exog
        self.gx = model.est[model.reg_type]
        self.test_vars = test_vars
        self.pivot = pivot
        self.run()

    def _compute_test_stat(self, Y, X):
        """
        Computes the test statistic.  See p.371 in [8].
        """
        pass

    def _compute_lambda(self, Y, X):
        """Computes only lambda -- the main part of the test statistic"""
        pass

    def _compute_se_lambda(self, Y, X):
        """
        Calculates the SE of lambda by nested resampling
        Used to pivot the statistic.
        Bootstrapping works better with estimating pivotal statistics
        but slows down computation significantly.
        """
        pass

    def _compute_sig(self):
        """
        Computes the significance value for the variable(s) tested.

        The empirical distribution of the test statistic is obtained through
        bootstrapping the sample.  The null hypothesis is rejected if the test
        statistic is larger than the 90, 95, 99 percentiles.
        """
        pass

class TestRegCoefD(TestRegCoefC):
    """
    Significance test for the categorical variables in a nonparametric
    regression.

    Parameters
    ----------
    model : Instance of KernelReg class
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars : tuple, list of one element
        index of position of the discrete variable to be tested
        for significance. E.g. (3) tests variable at
        position 3 for significance.
    nboot : int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400

    Attributes
    ----------
    sig : str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class currently does not allow joint hypothesis.
    Only one variable can be tested at a time

    References
    ----------
    See [9] and chapter 12 in [1].
    """

    def _compute_test_stat(self, Y, X):
        """Computes the test statistic"""
        pass

    def _compute_sig(self):
        """Calculates the significance level of the variable tested"""
        pass

    def _est_cond_mean(self):
        """
        Calculates the expected conditional mean
        m(X, Z=l) for all possible l
        """
        pass