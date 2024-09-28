import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils

def _calc_survfunc_right(time, status, weights=None, entry=None, compress=True, retall=True):
    """
    Calculate the survival function and its standard error for a single
    group.
    """
    pass

def _calc_incidence_right(time, status, weights=None):
    """
    Calculate the cumulative incidence function and its standard error.
    """
    pass

class CumIncidenceRight:
    """
    Estimation and inference for a cumulative incidence function.

    If J = 1, 2, ... indicates the event type, the cumulative
    incidence function for cause j is:

    I(t, j) = P(T <= t and J=j)

    Only right censoring is supported.  If frequency weights are provided,
    the point estimate is returned without a standard error.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        If status >= 1 indicates which event occurred at time t.  If
        status = 0, the subject was censored at time t.
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only
        used if exog is provided.
    dimred : bool
        If True, proportional hazards regression models are used to
        reduce exog to two columns by predicting overall events and
        censoring in two separate models.  If False, exog is used
        directly for calculating kernel weights without dimension
        reduction.

    Attributes
    ----------
    times : array_like
        The distinct times at which the incidence rates are estimated
    cinc : list of arrays
        cinc[k-1] contains the estimated cumulative incidence rates
        for outcome k=1,2,...
    cinc_se : list of arrays
        The standard errors for the values in `cinc`.  Not available when
        exog and/or frequency weights are provided.

    Notes
    -----
    When exog is provided, a local estimate of the cumulative incidence
    rate around each point is provided, and these are averaged to
    produce an estimate of the marginal cumulative incidence
    functions.  The procedure is analogous to that described in Zeng
    (2004) for estimation of the marginal survival function.  The
    approach removes bias resulting from dependent censoring when the
    censoring becomes independent conditioned on the columns of exog.

    References
    ----------
    The Stata stcompet procedure:
        http://www.stata-journal.com/sjpdf.html?articlenum=st0059

    Dinse, G. E. and M. G. Larson. 1986. A note on semi-Markov models
    for partially censored data. Biometrika 73: 379-386.

    Marubini, E. and M. G. Valsecchi. 1995. Analysing Survival Data
    from Clinical Trials and Observational Studies. Chichester, UK:
    John Wiley & Sons.

    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, title=None, freq_weights=None, exog=None, bw_factor=1.0, dimred=True):
        _checkargs(time, status, None, freq_weights, None)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        if exog is not None:
            from ._kernel_estimates import _kernel_cumincidence
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs ** (-1 / 3.0) * bw_factor
            kfunc = lambda x: np.exp(-x ** 2 / kw ** 2).sum(1)
            x = _kernel_cumincidence(time, status, exog, kfunc, freq_weights, dimred)
            self.times = x[0]
            self.cinc = x[1]
            return
        x = _calc_incidence_right(time, status, freq_weights)
        self.cinc = x[0]
        self.cinc_se = x[1]
        self.times = x[2]
        self.title = '' if not title else title

class SurvfuncRight:
    """
    Estimation and inference for a survival function.

    The survival function S(t) = P(T > t) is the probability that an
    event time T is greater than t.

    This class currently only supports right censoring.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        Status at the event time, status==1 is the 'event'
        (e.g. death, failure), meaning that the event
        occurs at the given value in `time`; status==0
        indicates that censoring has occurred, meaning that
        the event occurs after the given value in `time`.
    entry : array_like, optional An array of entry times for handling
        left truncation (the subject is not in the risk set on or
        before the entry time)
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only used
        if exog is provided.

    Attributes
    ----------
    surv_prob : array_like
        The estimated value of the survivor function at each time
        point in `surv_times`.
    surv_prob_se : array_like
        The standard errors for the values in `surv_prob`.  Not available
        if exog is provided.
    surv_times : array_like
        The points where the survival function changes.
    n_risk : array_like
        The number of subjects at risk just before each time value in
        `surv_times`.  Not available if exog is provided.
    n_events : array_like
        The number of events (e.g. deaths) that occur at each point
        in `surv_times`.  Not available if exog is provided.

    Notes
    -----
    If exog is None, the standard Kaplan-Meier estimator is used.  If
    exog is not None, a local estimate of the marginal survival
    function around each point is constructed, and these are then
    averaged.  This procedure gives an estimate of the marginal
    survival function that accounts for dependent censoring as long as
    the censoring becomes independent when conditioning on the
    covariates in exog.  See Zeng et al. (2004) for details.

    References
    ----------
    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, entry=None, title=None, freq_weights=None, exog=None, bw_factor=1.0):
        _checkargs(time, status, entry, freq_weights, exog)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        if entry is not None:
            entry = self.entry = np.asarray(entry)
        if exog is not None:
            if entry is not None:
                raise ValueError('exog and entry cannot both be present')
            from ._kernel_estimates import _kernel_survfunc
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs ** (-1 / 3.0) * bw_factor
            kfunc = lambda x: np.exp(-x ** 2 / kw ** 2).sum(1)
            x = _kernel_survfunc(time, status, exog, kfunc, freq_weights)
            self.surv_prob = x[0]
            self.surv_times = x[1]
            return
        x = _calc_survfunc_right(time, status, weights=freq_weights, entry=entry)
        self.surv_prob = x[0]
        self.surv_prob_se = x[1]
        self.surv_times = x[2]
        self.n_risk = x[4]
        self.n_events = x[5]
        self.title = '' if not title else title

    def plot(self, ax=None):
        """
        Plot the survival function.

        Examples
        --------
        Change the line color:

        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("flchain", "survival").data
        >>> df = data.loc[data.sex == "F", :]
        >>> sf = sm.SurvfuncRight(df["futime"], df["death"])
        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[0].set_color('purple')
        >>> li[1].set_color('purple')

        Do not show the censoring points:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[1].set_visible(False)
        """
        pass

    def quantile(self, p):
        """
        Estimated quantile of a survival distribution.

        Parameters
        ----------
        p : float
            The probability point at which the quantile
            is determined.

        Returns the estimated quantile.
        """
        pass

    def quantile_ci(self, p, alpha=0.05, method='cloglog'):
        """
        Returns a confidence interval for a survival quantile.

        Parameters
        ----------
        p : float
            The probability point for which a confidence interval is
            determined.
        alpha : float
            The confidence interval has nominal coverage probability
            1 - `alpha`.
        method : str
            Function to use for g-transformation, must be ...

        Returns
        -------
        lb : float
            The lower confidence limit.
        ub : float
            The upper confidence limit.

        Notes
        -----
        The confidence interval is obtained by inverting Z-tests.  The
        limits of the confidence interval will always be observed
        event times.

        References
        ----------
        The method is based on the approach used in SAS, documented here:

          http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm
        """
        pass

    def summary(self):
        """
        Return a summary of the estimated survival function.

        The summary is a dataframe containing the unique event times,
        estimated survival function values, and related quantities.
        """
        pass

    def simultaneous_cb(self, alpha=0.05, method='hw', transform='log'):
        """
        Returns a simultaneous confidence band for the survival function.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the desired simultaneous coverage
            probability for the confidence region.  Currently alpha
            must be set to 0.05, giving 95% simultaneous intervals.
        method : str
            The method used to produce the simultaneous confidence
            band.  Only the Hall-Wellner (hw) method is currently
            implemented.
        transform : str
            The used to produce the interval (note that the returned
            interval is on the survival probability scale regardless
            of which transform is used).  Only `log` and `arcsin` are
            implemented.

        Returns
        -------
        lcb : array_like
            The lower confidence limits corresponding to the points
            in `surv_times`.
        ucb : array_like
            The upper confidence limits corresponding to the points
            in `surv_times`.
        """
        pass

def survdiff(time, status, group, weight_type=None, strata=None, entry=None, **kwargs):
    """
    Test for the equality of two survival distributions.

    Parameters
    ----------
    time : array_like
        The event or censoring times.
    status : array_like
        The censoring status variable, status=1 indicates that the
        event occurred, status=0 indicates that the observation was
        censored.
    group : array_like
        Indicators of the two groups
    weight_type : str
        The following weight types are implemented:
            None (default) : logrank test
            fh : Fleming-Harrington, weights by S^(fh_p),
                 requires exponent fh_p to be provided as keyword
                 argument; the weights are derived from S defined at
                 the previous event time, and the first weight is
                 always 1.
            gb : Gehan-Breslow, weights by the number at risk
            tw : Tarone-Ware, weights by the square root of the number
                 at risk
    strata : array_like
        Optional stratum indicators for a stratified test
    entry : array_like
        Entry times to handle left truncation. The subject is not in
        the risk set on or before the entry time.

    Returns
    -------
    chisq : The chi-square (1 degree of freedom) distributed test
            statistic value
    pvalue : The p-value for the chi^2 test
    """
    pass

def plot_survfunc(survfuncs, ax=None):
    """
    Plot one or more survivor functions.

    Parameters
    ----------
    survfuncs : object or array_like
        A single SurvfuncRight object, or a list or SurvfuncRight
        objects that are plotted together.

    Returns
    -------
    A figure instance on which the plot was drawn.

    Examples
    --------
    Add a legend:

    >>> import statsmodels.api as sm
    >>> from statsmodels.duration.survfunc import plot_survfunc
    >>> data = sm.datasets.get_rdataset("flchain", "survival").data
    >>> df = data.loc[data.sex == "F", :]
    >>> sf0 = sm.SurvfuncRight(df["futime"], df["death"])
    >>> sf1 = sm.SurvfuncRight(3.0 * df["futime"], df["death"])
    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> leg = fig.legend((ha[0], ha[1]), (lb[0], lb[1]), loc='center right')

    Change the line colors:

    >>> fig = plot_survfunc([sf0, sf1])
    >>> ax = fig.get_axes()[0]
    >>> ax.set_position([0.1, 0.1, 0.64, 0.8])
    >>> ha, lb = ax.get_legend_handles_labels()
    >>> ha[0].set_color('purple')
    >>> ha[1].set_color('orange')
    """
    pass