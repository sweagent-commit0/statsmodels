"""Example for estimating distribution parameters when some are fixed.

This uses currently a patched version of the distributions, two methods are
added to the continuous distributions. This has no side effects.
It also adds bounds to vonmises, which changes the behavior of it for some
methods.

"""
import numpy as np
from scipy import stats
np.random.seed(12345)
x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)
print(stats.gamma.fit(x))
print(stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
print(stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2]))
np.random.seed(12345)
x = stats.lognorm.rvs(2, loc=0, scale=2, size=200)
print(stats.lognorm.fit_fr(x, frozen=[np.nan, 0.0, np.nan]))