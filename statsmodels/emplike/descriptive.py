"""
Empirical likelihood inference on descriptive statistics

This module conducts hypothesis tests and constructs confidence
intervals for the mean, variance, skewness, kurtosis and correlation.

If matplotlib is installed, this module can also generate multivariate
confidence region plots as well as mean-variance contour plots.

See _OptFuncts docstring for technical details and optimization variable
definitions.

General References:
------------------
Owen, A. (2001). "Empirical Likelihood." Chapman and Hall

"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils

def DescStat(endog):
    """
    Returns an instance to conduct inference on descriptive statistics
    via empirical likelihood.  See DescStatUV and DescStatMV for more
    information.

    Parameters
    ----------
    endog : ndarray
         Array of data

    Returns : DescStat instance
        If k=1, the function returns a univariate instance, DescStatUV.
        If k>1, the function returns a multivariate instance, DescStatMV.
    """
    pass

class _OptFuncts:
    """
    A class that holds functions that are optimized/solved.

    The general setup of the class is simple.  Any method that starts with
    _opt_ creates a vector of estimating equations named est_vect such that
    np.dot(p, (est_vect))=0 where p is the weight on each
    observation as a 1 x n array and est_vect is n x k.  Then _modif_Newton is
    called to determine the optimal p by solving for the Lagrange multiplier
    (eta) in the profile likelihood maximization problem.  In the presence
    of nuisance parameters, _opt_ functions are  optimized over to profile
    out the nuisance parameters.

    Any method starting with _ci_limits calculates the log likelihood
    ratio for a specific value of a parameter and then subtracts a
    pre-specified critical value.  This is solved so that llr - crit = 0.
    """

    def __init__(self, endog):
        pass

    def _log_star(self, eta, est_vect, weights, nobs):
        """
        Transforms the log of observation probabilities in terms of the
        Lagrange multiplier to the log 'star' of the probabilities.

        Parameters
        ----------
        eta : float
            Lagrange multiplier

        est_vect : ndarray (n,k)
            Estimating equations vector

        wts : nx1 array
            Observation weights

        Returns
        ------
        data_star : ndarray
            The weighted logstar of the estimting equations

        Notes
        -----
        This function is only a placeholder for the _fit_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computing the log likelihood ratio.
        """
        pass

    def _hess(self, eta, est_vect, weights, nobs):
        """
        Calculates the hessian of a weighted empirical likelihood
        problem.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        hess : m x m array
            Weighted hessian used in _wtd_modif_newton
        """
        pass

    def _grad(self, eta, est_vect, weights, nobs):
        """
        Calculates the gradient of a weighted empirical likelihood
        problem

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        gradient : ndarray (m,1)
            The gradient used in _wtd_modif_newton
        """
        pass

    def _modif_newton(self, eta, est_vect, weights):
        """
        Modified Newton's method for maximizing the log 'star' equation.  This
        function calls _fit_newton to find the optimal values of eta.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        params : 1xm array
            Lagrange multiplier that maximizes the log-likelihood
        """
        pass

    def _find_eta(self, eta):
        """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
        eta when computing ELR for univariate mean.

        Parameters
        ----------
        eta : float
            Lagrange multiplier in the empirical likelihood maximization

        Returns
        -------
        llr : float
            n times the log likelihood value for a given value of eta
        """
        pass

    def _ci_limits_mu(self, mu):
        """
        Calculates the difference between the log likelihood of mu_test and a
        specified critical value.

        Parameters
        ----------
        mu : float
           Hypothesized value of the mean.

        Returns
        -------
        diff : float
            The difference between the log likelihood value of mu0 and
            a specified value.
        """
        pass

    def _find_gamma(self, gamma):
        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean

        Parameters
        ----------
        gamma : float
            Lagrange multiplier when computing confidence interval

        Returns
        -------
        diff : float
            The difference between the log-liklihood when the Lagrange
            multiplier is gamma and a pre-specified value
        """
        pass

    def _opt_var(self, nuisance_mu, pval=False):
        """
        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance

        Parameters
        ----------
        nuisance_mu : float
            Value of a nuisance mean parameter

        Returns
        -------
        llr : float
            Log likelihood of a pre-specified variance holding the nuisance
            parameter constant
        """
        pass

    def _ci_limits_var(self, var):
        """
        Used to determine the confidence intervals for the variance.
        It calls test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        Parameters
        ----------
        var_test : float
            Hypothesized value of the variance

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at var_test and a
            pre-specified value.
        """
        pass

    def _opt_skew(self, nuis_params):
        """
        Called by test_skew.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a  nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-specified skewness holding
            the nuisance parameters constant.
        """
        pass

    def _opt_kurt(self, nuis_params):
        """
        Called by test_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-speified kurtosis holding the
            nuisance parameters constant
        """
        pass

    def _opt_skew_kurt(self, nuis_params):
        """
        Called by test_joint_skew_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        ------
        llr : float
            The log likelihood ratio of a pre-speified skewness and
            kurtosis holding the nuisance parameters constant.
        """
        pass

    def _ci_limits_skew(self, skew):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of skewness

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at skew and a
            pre-specified value.
        """
        pass

    def _ci_limits_kurt(self, kurt):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of kurtosis

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at kurt and a
            pre-specified value.
        """
        pass

    def _opt_correl(self, nuis_params, corr0, endog, nobs, x0, weights0):
        """
        Parameters
        ----------
        nuis_params : 1darray
            Array containing two nuisance means and two nuisance variances

        Returns
        -------
        llr : float
            The log-likelihood of the correlation coefficient holding nuisance
            parameters constant
        """
        pass

class DescStatUV(_OptFuncts):
    """
    A class to compute confidence intervals and hypothesis tests involving
    mean, variance, kurtosis and skewness of a univariate random variable.

    Parameters
    ----------
    endog : 1darray
        Data to be analyzed

    Attributes
    ----------
    endog : 1darray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = np.squeeze(endog)
        self.nobs = endog.shape[0]

    def test_mean(self, mu0, return_weights=False):
        """
        Returns - 2 x log-likelihood ratio, p-value and weights
        for a hypothesis test of the mean.

        Parameters
        ----------
        mu0 : float
            Mean value to be tested

        return_weights : bool
            If return_weights is True the function returns
            the weights of the observations under the null hypothesis.
            Default is False

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of mu0
        """
        pass

    def ci_mean(self, sig=0.05, method='gamma', epsilon=10 ** (-8), gamma_low=-10 ** 10, gamma_high=10 ** 10):
        """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig : float
            significance level. Default is .05

        method : str
            Root finding method,  Can be 'nested-brent' or
            'gamma'.  Default is 'gamma'

            'gamma' Tries to solve for the gamma parameter in the
            Lagrange (see Owen pg 22) and then determine the weights.

            'nested brent' uses brents method to find the confidence
            intervals but must maximize the likelihood ratio on every
            iteration.

            gamma is generally much faster.  If the optimizations does not
            converge, try expanding the gamma_high and gamma_low
            variable.

        gamma_low : float
            Lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low.

        gamma_high : float
            Upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high.

        epsilon : float
            When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ``**`` -6) consider shrinking epsilon

            When using 'gamma', amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If function returns f(a) and f(b) must have different signs,
            consider lowering epsilon.

        Returns
        -------
        Interval : tuple
            Confidence interval for the mean
        """
        pass

    def test_var(self, sig2_0, return_weights=False):
        """
        Returns  -2 x log-likelihood ratio and the p-value for the
        hypothesized variance

        Parameters
        ----------
        sig2_0 : float
            Hypothesized variance to be tested

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of observing sig2_0. Default is False

        Returns
        -------
        test_results : tuple
            The  log-likelihood ratio and the p_value  of sig2_0

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(1000)*100
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> hyp_test = el_analysis.test_var(9500)
        """
        pass

    def ci_var(self, lower_bound=None, upper_bound=None, sig=0.05):
        """
        Returns the confidence interval for the variance.

        Parameters
        ----------
        lower_bound : float
            The minimum value the lower confidence interval can
            take. The p-value from test_var(lower_bound) must be lower
            than 1 - significance level. Default is .99 confidence
            limit assuming normality

        upper_bound : float
            The maximum value the upper confidence interval
            can take. The p-value from test_var(upper_bound) must be lower
            than 1 - significance level.  Default is .99 confidence
            limit assuming normality

        sig : float
            The significance level. Default is .05

        Returns
        -------
        Interval : tuple
            Confidence interval for the variance

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(100)
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> el_analysis.ci_var()
        (0.7539322567470305, 1.229998852496268)
        >>> el_analysis.ci_var(.5, 2)
        (0.7539322567469926, 1.2299988524962664)

        Notes
        -----
        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering lower_bound and raising
        upper_bound.
        """
        pass

    def plot_contour(self, mu_low, mu_high, var_low, var_high, mu_step, var_step, levs=[0.2, 0.1, 0.05, 0.01, 0.001]):
        """
        Returns a plot of the confidence region for a univariate
        mean and variance.

        Parameters
        ----------
        mu_low : float
            Lowest value of the mean to plot

        mu_high : float
            Highest value of the mean to plot

        var_low : float
            Lowest value of the variance to plot

        var_high : float
            Highest value of the variance to plot

        mu_step : float
            Increments to evaluate the mean

        var_step : float
            Increments to evaluate the mean

        levs : list
            Which values of significance the contour lines will be drawn.
            Default is [.2, .1, .05, .01, .001]

        Returns
        -------
        Figure
            The contour plot
        """
        pass

    def test_skew(self, skew0, return_weights=False):
        """
        Returns  -2 x log-likelihood and p-value for the hypothesized
        skewness.

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p_value of skew0
        """
        pass

    def test_kurt(self, kurt0, return_weights=False):
        """
        Returns -2 x log-likelihood and the p-value for the hypothesized
        kurtosis.

        Parameters
        ----------
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of kurt0
        """
        pass

    def test_joint_skew_kurt(self, skew0, kurt0, return_weights=False):
        """
        Returns - 2 x log-likelihood and the p-value for the joint
        hypothesis test for skewness and kurtosis

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value  of the joint hypothesis test.
        """
        pass

    def ci_skew(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for skewness.

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of skewness the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of skewness the lower limit can be.
            Default is .99 confidence level assuming normality.

        Returns
        -------
        Interval : tuple
            Confidence interval for the skewness

        Notes
        -----
        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bounds
        """
        pass

    def ci_kurt(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for kurtosis.

        Parameters
        ----------

        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of kurtosis the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of kurtosis the lower limit can be.
            Default is .99 confidence limit assuming normality.

        Returns
        -------
        Interval : tuple
            Lower and upper confidence limit

        Notes
        -----
        For small n, upper_bound and lower_bound may have to be
        provided by the user.  Consider using test_kurt to find
        values close to the desired significance level.

        If function returns f(a) and f(b) must have different signs, consider
        expanding the bounds.
        """
        pass

class DescStatMV(_OptFuncts):
    """
    A class for conducting inference on multivariate means and correlation.

    Parameters
    ----------
    endog : ndarray
        Data to be analyzed

    Attributes
    ----------
    endog : ndarray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = endog
        self.nobs = endog.shape[0]

    def mv_test_mean(self, mu_array, return_weights=False):
        """
        Returns -2 x log likelihood and the p-value
        for a multivariate hypothesis test of the mean

        Parameters
        ----------
        mu_array  : 1d array
            Hypothesized values for the mean.  Must have same number of
            elements as columns in endog

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of mu_array. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value for mu_array
        """
        pass

    def mv_mean_contour(self, mu1_low, mu1_upp, mu2_low, mu2_upp, step1, step2, levs=(0.001, 0.01, 0.05, 0.1, 0.2), var1_name=None, var2_name=None, plot_dta=False):
        """
        Creates a confidence region plot for the mean of bivariate data

        Parameters
        ----------
        m1_low : float
            Minimum value of the mean for variable 1
        m1_upp : float
            Maximum value of the mean for variable 1
        mu2_low : float
            Minimum value of the mean for variable 2
        mu2_upp : float
            Maximum value of the mean for variable 2
        step1 : float
            Increment of evaluations for variable 1
        step2 : float
            Increment of evaluations for variable 2
        levs : list
            Levels to be drawn on the contour plot.
            Default =  (.001, .01, .05, .1, .2)
        plot_dta : bool
            If True, makes a scatter plot of the data on
            top of the contour plot. Defaultis False.
        var1_name : str
            Name of variable 1 to be plotted on the x-axis
        var2_name : str
            Name of variable 2 to be plotted on the y-axis

        Notes
        -----
        The smaller the step size, the more accurate the intervals
        will be

        If the function returns optimization failed, consider narrowing
        the boundaries of the plot

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> two_rvs = np.random.standard_normal((20,2))
        >>> el_analysis = sm.emplike.DescStat(two_rvs)
        >>> contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)
        >>> contourp.show()
        """
        pass

    def test_corr(self, corr0, return_weights=0):
        """
        Returns -2 x log-likelihood ratio and  p-value for the
        correlation coefficient between 2 variables

        Parameters
        ----------
        corr0 : float
            Hypothesized value to be tested

        return_weights : bool
            If true, returns the weights that maximize
            the log-likelihood at the hypothesized value
        """
        pass

    def ci_corr(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence intervals for the correlation coefficient

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value the upper confidence limit can be.
            Default is  99% confidence limit assuming normality.

        lower_bound : float
            Minimum value the lower confidence limit can be.
            Default is 99% confidence limit assuming normality.

        Returns
        -------
        interval : tuple
            Confidence interval for the correlation
        """
        pass