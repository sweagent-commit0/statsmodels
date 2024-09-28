"""
Created on Sun Aug 01 19:20:16 2010

Author: josef-pktd
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
nobs = 1000
r = stats.pareto.rvs(1, size=nobs)
rhisto, e = np.histogram(np.clip(r, 0, 1000), bins=50)
plt.figure()
plt.loglog(e[:-1] + np.diff(e) / 2, rhisto, '-o')
plt.figure()
plt.loglog(e[:-1] + np.diff(e) / 2, nobs - rhisto.cumsum(), '-o')
rsind = np.argsort(r)
rs = r[rsind]
rsf = nobs - rsind.argsort()
plt.figure()
plt.loglog(rs, nobs - np.arange(nobs), '-o')
print(stats.linregress(np.log(rs), np.log(nobs - np.arange(nobs))))
plt.show()