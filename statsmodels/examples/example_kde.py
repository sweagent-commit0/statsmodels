import numpy as np
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import kdensityfft
from statsmodels.nonparametric import bandwidths
import matplotlib.pyplot as plt
np.random.seed(12345)
obs_dist = mixture_rvs([0.25, 0.75], size=10000, dist=[stats.norm, stats.norm], kwargs=(dict(loc=-1, scale=0.5), dict(loc=1, scale=0.5)))
f_hat, grid, bw = kdensityfft(obs_dist, kernel='gauss', bw='scott')
plt.figure()
plt.hist(obs_dist, bins=50, normed=True, color='red')
plt.plot(grid, f_hat, lw=2, color='black')
plt.show()
bw = bandwidths.bw_scott(obs_dist)