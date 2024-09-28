"""example for grid of scatter plots with probability ellipses


Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lrange
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.plot_grids import scatter_ellipse
nvars = 6
mmean = np.arange(1.0, nvars + 1) / nvars * 1.5
rho = 0.5
r = np.random.uniform(-0.99, 0.99, size=(nvars, nvars))
r = (r + r.T) / 2.0
assert np.allclose(r, r.T)
mcorr = r
mcorr[lrange(nvars), lrange(nvars)] = 1
mstd = np.arange(1.0, nvars + 1) / nvars
mcov = mcorr * np.outer(mstd, mstd)
evals = np.linalg.eigvalsh(mcov)
assert evals.min > 0
nobs = 100
data = np.random.multivariate_normal(mmean, mcov, size=nobs)
dmean = data.mean(0)
dcov = np.cov(data, rowvar=0)
print(dmean)
print(dcov)
dcorr = np.corrcoef(data, rowvar=0)
dcorr[np.triu_indices(nvars)] = 0
print(dcorr)
varnames = ['var%d' % i for i in range(nvars)]
fig = scatter_ellipse(data, level=0.9, varnames=varnames)
plt.show()