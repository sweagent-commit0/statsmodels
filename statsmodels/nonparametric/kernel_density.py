"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types.

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
import numpy as np
from . import kernels
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, LeaveOneOut, _adjust_shape
__all__ = ['KDEMultivariate', 'KDEMultivariateConditional', 'EstimatorSettings']

class KDEMultivariate(GenericKDE):
    """
    Multivariate kernel density estimator.

    This density estimator can handle univariate as well as multivariate data,
    including mixed continuous / ordered discrete / unordered discrete data.
    It also provides cross-validated bandwidth selection methods (least
    squares, maximum likelihood).

    Parameters
    ----------
    data : list of ndarrays or 2-D ndarray
        The training data for the Kernel Density Estimation, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    var_type : str
        The type of the variables:

            - c : continuous
            - u : unordered (discrete)
            - o : ordered (discrete)

        The string should contain a type specifier for each variable, so for
        example ``var_type='ccuo'``.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults : EstimatorSettings instance, optional
        The default values for (efficient) bandwidth estimation.

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters.

    See Also
    --------
    KDEMultivariateConditional

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2, 1, size=(nobs,1))

    Estimate a bivariate distribution and display the bandwidth found:

    >>> dens_u = sm.nonparametric.KDEMultivariate(data=[c1,c2],
    ...     var_type='cc', bw='normal_reference')
    >>> dens_u.bw
    array([ 0.39967419,  0.38423292])
    """

    def __init__(self, data, var_type, bw=None, defaults=None):
        self.var_type = var_type
        self.k_vars = len(self.var_type)
        self.data = _adjust_shape(data, self.k_vars)
        self.data_type = var_type
        self.nobs, self.k_vars = np.shape(self.data)
        if self.nobs <= self.k_vars:
            raise ValueError('The number of observations must be larger than the number of variables.')
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KDE instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        return rpr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out likelihood function.

        The leave-one-out likelihood function for the unconditional KDE.

        Parameters
        ----------
        bw : array_like
            The value for the bandwidth parameter(s).
        func : callable, optional
            Function to transform the likelihood values (before summing); for
            the log likelihood, use ``func=np.log``.  Default is ``f(x) = x``.

        Notes
        -----
        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\\frac{1}{(n-1)h}
                    \\sum_{j=1,j\\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \\prod_{s=1}^{q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
        pass

    def pdf(self, data_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        data_predict : array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        pdf_est : array_like
            Probability density function evaluated at `data_predict`.

        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \\prod_{s=1}^{q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
        pass

    def cdf(self, data_predict=None):
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        data_predict : array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        cdf_est : array_like
            The estimate of the cdf.

        Notes
        -----
        See https://en.wikipedia.org/wiki/Cumulative_distribution_function
        For more details on the estimation see Ref. [5] in module docstring.

        The multivariate CDF for mixed data (continuous and ordered/unordered
        discrete) is estimated by:

        .. math::

            F(x^{c},x^{d})=n^{-1}\\sum_{i=1}^{n}\\left[G(\\frac{x^{c}-X_{i}}{h})\\sum_{u\\leq x^{d}}L(X_{i}^{d},x_{i}^{d}, \\lambda)\\right]

        where G() is the product kernel CDF estimator for the continuous
        and L() for the discrete variables.

        Used bandwidth is ``self.bw``.
        """
        pass

    def imse(self, bw):
        """
        Returns the Integrated Mean Square Error for the unconditional KDE.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV : float
            The cross-validation objective function.

        Notes
        -----
        See p. 27 in [1]_ for details on how to handle the multivariate
        estimation with mixed data types see p.6 in [2]_.

        The formula for the cross-validation objective function is:

        .. math:: CV=\\frac{1}{n^{2}}\\sum_{i=1}^{n}\\sum_{j=1}^{N}
            \\bar{K}_{h}(X_{i},X_{j})-\\frac{2}{n(n-1)}\\sum_{i=1}^{n}
            \\sum_{j=1,j\\neq i}^{N}K_{h}(X_{i},X_{j})

        Where :math:`\\bar{K}_{h}` is the multivariate product convolution
        kernel (consult [2]_ for mixed data types).

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
                with Categorical and Continuous Data." Working Paper. (2000)
        """
        pass

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        pass

class KDEMultivariateConditional(GenericKDE):
    """
    Conditional multivariate kernel density estimator.

    Calculates ``P(Y_1,Y_2,...Y_n | X_1,X_2...X_m) =
    P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(X_1, X_2,..., X_m)``.
    The conditional density is by definition the ratio of the two densities,
    see [1]_.

    Parameters
    ----------
    endog : list of ndarrays or 2-D ndarray
        The training data for the dependent variables, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    exog : list of ndarrays or 2-D ndarray
        The training data for the independent variable; same shape as `endog`.
    dep_type : str
        The type of the dependent variables:

            c : Continuous
            u : Unordered (Discrete)
            o : Ordered (Discrete)

        The string should contain a type specifier for each variable, so for
        example ``dep_type='ccuo'``.
    indep_type : str
        The type of the independent variables; specified like `dep_type`.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults : Instance of class EstimatorSettings
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters

    See Also
    --------
    KDEMultivariate

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conditional_probability_distribution

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> nobs = 300
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2,1,size=(nobs,1))

    >>> dens_c = sm.nonparametric.KDEMultivariateConditional(endog=[c1],
    ...     exog=[c2], dep_type='c', indep_type='c', bw='normal_reference')
    >>> dens_c.bw   # show computed bandwidth
    array([ 0.41223484,  0.40976931])
    """

    def __init__(self, endog, exog, dep_type, indep_type, bw, defaults=None):
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.data_type = dep_type + indep_type
        self.k_dep = len(self.dep_type)
        self.k_indep = len(self.indep_type)
        self.endog = _adjust_shape(endog, self.k_dep)
        self.exog = _adjust_shape(exog, self.k_indep)
        self.nobs, self.k_dep = np.shape(self.endog)
        self.data = np.column_stack((self.endog, self.exog))
        self.k_vars = np.shape(self.data)[1]
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KDEMultivariateConditional instance\n'
        rpr += 'Number of independent variables: k_indep = ' + str(self.k_indep) + '\n'
        rpr += 'Number of dependent variables: k_dep = ' + str(self.k_dep) + '\n'
        rpr += 'Number of observations: nobs = ' + str(self.nobs) + '\n'
        rpr += 'Independent variable types:      ' + self.indep_type + '\n'
        rpr += 'Dependent variable types:      ' + self.dep_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        return rpr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out conditional likelihood of the data.

        If `func` is not equal to the default, what's calculated is a function
        of the leave-one-out conditional likelihood.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).
        func : callable, optional
            Function to transform the likelihood values (before summing); for
            the log likelihood, use ``func=np.log``.  Default is ``f(x) = x``.

        Returns
        -------
        L : float
            The value of the leave-one-out function for the data.

        Notes
        -----
        Similar to ``KDE.loo_likelihood`, but substitute ``f(y|x)=f(x,y)/f(x)``
        for ``f(x)``.
        """
        pass

    def pdf(self, endog_predict=None, exog_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        endog_predict : array_like, optional
            Evaluation data for the dependent variables.  If unspecified, the
            training data is used.
        exog_predict : array_like, optional
            Evaluation data for the independent variables.

        Returns
        -------
        pdf : array_like
            The value of the probability density at `endog_predict` and `exog_predict`.

        Notes
        -----
        The formula for the conditional probability density is:

        .. math:: f(y|x)=\\frac{f(x,y)}{f(x)}

        with

        .. math:: f(x)=\\prod_{s=1}^{q}h_{s}^{-1}k
                            \\left(\\frac{x_{is}-x_{js}}{h_{s}}\\right)

        where :math:`k` is the appropriate kernel for each variable.
        """
        pass

    def cdf(self, endog_predict=None, exog_predict=None):
        """
        Cumulative distribution function for the conditional density.

        Parameters
        ----------
        endog_predict : array_like, optional
            The evaluation dependent variables at which the cdf is estimated.
            If not specified the training dependent variables are used.
        exog_predict : array_like, optional
            The evaluation independent variables at which the cdf is estimated.
            If not specified the training independent variables are used.

        Returns
        -------
        cdf_est : array_like
            The estimate of the cdf.

        Notes
        -----
        For more details on the estimation see [2]_, and p.181 in [1]_.

        The multivariate conditional CDF for mixed data (continuous and
        ordered/unordered discrete) is estimated by:

        .. math::

            F(y|x)=\\frac{n^{-1}\\sum_{i=1}^{n}G(\\frac{y-Y_{i}}{h_{0}}) W_{h}(X_{i},x)}{\\widehat{\\mu}(x)}

        where G() is the product kernel CDF estimator for the dependent (y)
        variable(s) and W() is the product kernel CDF estimator for the
        independent variable(s).

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Liu, R., Yang, L. "Kernel estimation of multivariate cumulative
                    distribution function." Journal of Nonparametric
                    Statistics (2008)
        """
        pass

    def imse(self, bw):
        """
        The integrated mean square error for the conditional KDE.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV : float
            The cross-validation objective function.

        Notes
        -----
        For more details see pp. 156-166 in [1]_. For details on how to
        handle the mixed variable types see [2]_.

        The formula for the cross-validation objective function for mixed
        variable types is:

        .. math:: CV(h,\\lambda)=\\frac{1}{n}\\sum_{l=1}^{n}
            \\frac{G_{-l}(X_{l})}{\\left[\\mu_{-l}(X_{l})\\right]^{2}}-
            \\frac{2}{n}\\sum_{l=1}^{n}\\frac{f_{-l}(X_{l},Y_{l})}{\\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l}) = n^{-2}\\sum_{i\\neq l}\\sum_{j\\neq l}
                        K_{X_{i},X_{l}} K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel and
        :math:`\\mu_{-l}(X_{l})` is the leave-one-out estimator of the pdf.

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel.

        The value of the function is minimized by the ``_cv_ls`` method of the
        `GenericKDE` class to return the bw estimates that minimize the
        distance between the estimated and "true" probability density.

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
                with Categorical and Continuous Data." Working Paper. (2000)
        """
        pass

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        pass