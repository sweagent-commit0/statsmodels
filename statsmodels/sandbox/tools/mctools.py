"""Helper class for Monte Carlo Studies for (currently) statistical tests

Most of it should also be usable for Bootstrap, and for MC for estimators.
Takes the sample generator, dgb, and the statistical results, statistic,
as functions in the argument.


Author: Josef Perktold (josef-pktd)
License: BSD-3


TODOs, Design
-------------
If we only care about univariate analysis, i.e. marginal if statistics returns
more than one value, the we only need to store the sorted mcres not the
original res. Do we want to extend to multivariate analysis?

Use distribution function to keep track of MC results, ECDF, non-paramatric?
Large parts are similar to a 2d array of independent multivariate random
variables. Joint distribution is not used (yet).

I guess this is currently only for one sided test statistics, e.g. for
two-sided tests basend on t or normal distribution use the absolute value.

"""
from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable

class StatTestMC:
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print(summary, for quantiles and for histogram
    draft in trying out script log

    Parameters
    ----------
    dgp : callable
        Function that generates the data to be used in Monte Carlo that should
        return a new sample with each call
    statistic : callable
        Function that calculates the test statistic, which can return either
        a single statistic or a 1d array_like (tuple, list, ndarray).
        see also statindices in description of run

    Attributes
    ----------
    many methods store intermediate results

    self.mcres : ndarray (nrepl, nreturns) or (nrepl, len(statindices))
        Monte Carlo results stored by run


    Notes
    -----

    .. Warning::
       This is (currently) designed for a single call to run. If run is
       called a second time with different arguments, then some attributes might
       not be updated, and, therefore, not correspond to the same run.

    .. Warning::
       Under Construction, do not expect stability in Api or implementation


    Examples
    --------

    Define a function that defines our test statistic:

    def lb(x):
        s,p = acorr_ljungbox(x, lags=4)
        return np.r_[s, p]

    Note lb returns eight values.

    Define a random sample generator, for example 500 independently, normal
    distributed observations in a sample:


    def normalnoisesim(nobs=500, loc=0.0):
        return (loc+np.random.randn(nobs))

    Create instance and run Monte Carlo. Using statindices=list(range(4)) means that
    only the first for values of the return of the statistic (lb) are stored
    in the Monte Carlo results.

    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(5000, statindices=list(range(4)))

    Most of the other methods take an idx which indicates for which columns
    the results should be presented, e.g.

    print(mc1.cdf(crit, [1,2,3])[1]
    """

    def __init__(self, dgp, statistic):
        self.dgp = dgp
        self.statistic = statistic

    def run(self, nrepl, statindices=None, dgpargs=[], statsargs=[]):
        """run the actual Monte Carlo and save results

        Parameters
        ----------
        nrepl : int
            number of Monte Carlo repetitions
        statindices : None or list of integers
           determines which values of the return of the statistic
           functions are stored in the Monte Carlo. Default None
           means the entire return. If statindices is a list of
           integers, then it will be used as index into the return.
        dgpargs : tuple
           optional parameters for the DGP
        statsargs : tuple
           optional parameters for the statistics function

        Returns
        -------
        None, all results are attached


        """
        pass

    def histogram(self, idx=None, critval=None):
        """calculate histogram values

        does not do any plotting

        I do not remember what I wanted here, looks similar to the new cdf
        method, but this also does a binned pdf (self.histo)


        """
        pass

    def quantiles(self, idx=None, frac=[0.01, 0.025, 0.05, 0.1, 0.975]):
        """calculate quantiles of Monte Carlo results

        similar to ppf

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            Defines which quantiles should be calculated. For example a frac
            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1

        Returns
        -------
        frac : ndarray
            same values as input, TODO: I should drop this again ?
        quantiles : ndarray, (len(frac), len(idx))
            the quantiles with frac in rows and idx variables in columns

        Notes
        -----

        rename to ppf ? make frac required
        change sequence idx, frac


        """
        pass

    def cdf(self, x, idx=None):
        """calculate cumulative probabilities of Monte Carlo results

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            Defines which quantiles should be calculated. For example a frac
            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1

        Returns
        -------
        x : ndarray
            same as input, TODO: I should drop this again ?
        probs : ndarray, (len(x), len(idx))
            the quantiles with frac in rows and idx variables in columns



        """
        pass

    def plot_hist(self, idx, distpdf=None, bins=50, ax=None, kwds=None):
        """plot the histogram against a reference distribution

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        distpdf : callable
            probability density function of reference distribution
        bins : {int, array_like}
            used unchanged for matplotlibs hist call
        ax : TODO: not implemented yet
        kwds : None or tuple of dicts
            extra keyword options to the calls to the matplotlib functions,
            first dictionary is for his, second dictionary for plot of the
            reference distribution

        Returns
        -------
        None


        """
        pass

    def summary_quantiles(self, idx, distppf, frac=[0.01, 0.025, 0.05, 0.1, 0.975], varnames=None, title=None):
        """summary table for quantiles (critical values)

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        distppf : callable
            probability density function of reference distribution
            TODO: use `crit` values instead or additional, see summary_cdf
        frac : array_like, float
            probabilities for which
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results

        """
        pass

    def summary_cdf(self, idx, frac, crit, varnames=None, title=None):
        """summary table for cumulative density function


        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            probabilities for which
        crit : array_like
            values for which cdf is calculated
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results


        """
        pass
if __name__ == '__main__':
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print('\nLjung Box')
    print('Results with MC class')
    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(10000, statindices=lrange(8))
    print(mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975]))
    print(mc1.quantiles(1))
    print(mc1.quantiles(0))
    print(mc1.histogram(0))
    print(mc1.summary_quantiles([1, 2, 3], stats.chi2([2, 3, 4]).ppf, varnames=['lag 1', 'lag 2', 'lag 3'], title='acorr_ljungbox'))
    print(mc1.cdf(0.1026, 1))
    print(mc1.cdf(0.7278, 3))
    print(mc1.cdf(0.7278, [1, 2, 3]))
    frac = [0.01, 0.025, 0.05, 0.1, 0.975]
    crit = stats.chi2([2, 4]).ppf(np.atleast_2d(frac).T)
    print(mc1.summary_cdf([1, 3], frac, crit, title='acorr_ljungbox'))
    crit = stats.chi2([2, 3, 4]).ppf(np.atleast_2d(frac).T)
    print(mc1.summary_cdf([1, 2, 3], frac, crit, varnames=['lag 1', 'lag 2', 'lag 3'], title='acorr_ljungbox'))
    print(mc1.cdf(crit, [1, 2, 3])[1].shape)
    '\n    >>> mc1.cdf(crit[:,0], [1])[1].shape\n    (5, 1)\n    >>> mc1.cdf(crit[:,0], [1,3])[1].shape\n    (5, 2)\n    >>> mc1.cdf(crit[:,:], [1,3])[1].shape\n    (2, 5, 2)\n    '
    doplot = 0
    if doplot:
        import matplotlib.pyplot as plt
        mc1.plot_hist(0, stats.chi2(2).pdf)
        plt.show()