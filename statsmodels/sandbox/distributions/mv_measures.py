"""using multivariate dependence and divergence measures

The standard correlation coefficient measures only linear dependence between
random variables.
kendall's tau measures any monotonic relationship also non-linear.

mutual information measures any kind of dependence, but does not distinguish
between positive and negative relationship


mutualinfo_kde and mutualinfo_binning follow Khan et al. 2007

Shiraj Khan, Sharba Bandyopadhyay, Auroop R. Ganguly, Sunil Saigal,
David J. Erickson, III, Vladimir Protopopescu, and George Ostrouchov,
Relative performance of mutual information estimation methods for
quantifying the dependence among short and noisy data,
Phys. Rev. E 76, 026209 (2007)
http://pre.aps.org/abstract/PRE/v76/i2/e026209


"""
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.sandbox.infotheo as infotheo

def mutualinfo_kde(y, x, normed=True):
    """mutual information of two random variables estimated with kde

    """
    pass

def mutualinfo_kde_2sample(y, x, normed=True):
    """mutual information of two random variables estimated with kde

    """
    pass

def mutualinfo_binned(y, x, bins, normed=True):
    """mutual information of two random variables estimated with kde



    Notes
    -----
    bins='auto' selects the number of bins so that approximately 5 observations
    are expected to be in each bin under the assumption of independence. This
    follows roughly the description in Kahn et al. 2007

    """
    pass
if __name__ == '__main__':
    import statsmodels.api as sm
    funtype = ['linear', 'quadratic'][1]
    nobs = 200
    sig = 2
    x = np.sort(3 * np.random.randn(nobs))
    exog = sm.add_constant(x, prepend=True)
    if funtype == 'quadratic':
        y = 0 + x ** 2 + sig * np.random.randn(nobs)
    if funtype == 'linear':
        y = 0 + x + sig * np.random.randn(nobs)
    print('correlation')
    print(np.corrcoef(y, x)[0, 1])
    print('pearsonr', stats.pearsonr(y, x))
    print('spearmanr', stats.spearmanr(y, x))
    print('kendalltau', stats.kendalltau(y, x))
    pxy, binsx, binsy = np.histogram2d(x, y, bins=5)
    px, binsx_ = np.histogram(x, bins=binsx)
    py, binsy_ = np.histogram(y, bins=binsy)
    print('mutualinfo', infotheo.mutualinfo(px * 1.0 / nobs, py * 1.0 / nobs, 1e-15 + pxy * 1.0 / nobs, logbase=np.e))
    print('mutualinfo_kde normed', mutualinfo_kde(y, x))
    print('mutualinfo_kde       ', mutualinfo_kde(y, x, normed=False))
    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = mutualinfo_binned(y, x, 5, normed=True)
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())
    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = mutualinfo_binned(y, x, 'auto', normed=True)
    print('auto')
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())
    ys = np.sort(y)
    xs = np.sort(x)
    by = ys[((nobs - 1) * np.array([0, 0.25, 0.4, 0.6, 0.75, 1])).astype(int)]
    bx = xs[((nobs - 1) * np.array([0, 0.25, 0.4, 0.6, 0.75, 1])).astype(int)]
    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = mutualinfo_binned(y, x, (by, bx), normed=True)
    print('quantiles')
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())
    doplot = 1
    if doplot:
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'o')
        olsres = sm.OLS(y, exog).fit()
        plt.plot(x, olsres.fittedvalues)