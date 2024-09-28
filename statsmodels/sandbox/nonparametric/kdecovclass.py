"""subclassing kde

Author: josef pktd
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt

class gaussian_kde_set_covariance(stats.gaussian_kde):
    """
    from Anne Archibald in mailinglist:
    http://www.nabble.com/Width-of-the-gaussian-in-stats.kde.gaussian_kde---td19558924.html#a19558924
    """

    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)

class gaussian_kde_covfact(stats.gaussian_kde):

    def __init__(self, dataset, covfact='scotts'):
        self.covfact = covfact
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance_(self):
        """not used"""
        pass
if __name__ == '__main__':
    n_basesample = 1000
    np.random.seed(8765678)
    alpha = 0.6
    mlow, mhigh = (-3, 3)
    xn = np.concatenate([mlow + np.random.randn(alpha * n_basesample), mhigh + np.random.randn((1 - alpha) * n_basesample)])
    gkde = gaussian_kde_covfact(xn, 0.1)
    ind = np.linspace(-7, 7, 101)
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    plt.hist(xn, bins=20, normed=1)
    plt.plot(ind, kdepdf, label='kde', color='g')
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) + (1 - alpha) * stats.norm.pdf(ind, loc=mhigh), color='r', label='DGP: normal mix')
    plt.title('Kernel Density Estimation')
    plt.legend()
    gkde = gaussian_kde_covfact(xn, 'scotts')
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    plt.hist(xn, bins=20, normed=1)
    plt.plot(ind, kdepdf, label='kde', color='g')
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) + (1 - alpha) * stats.norm.pdf(ind, loc=mhigh), color='r', label='DGP: normal mix')
    plt.title('Kernel Density Estimation')
    plt.legend()
    for cv in ['scotts', 'silverman', 0.05, 0.1, 0.5]:
        plotkde(cv)
    test_kde_1d()
    np.random.seed(8765678)
    n_basesample = 1000
    xn = np.random.randn(n_basesample)
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)
    gkde = stats.gaussian_kde(xn)