"""Plot acf and pacf for some ARMA(1,1)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.tsa.arima_process as tsp
from statsmodels.graphics.tsaplots import plotacf
np.set_printoptions(precision=2)
arcoefs = [0.9, 0.0, -0.5]
macoefs = [0.9, 0.0, -0.5]
nsample = 1000
nburnin = 1000
sig = 1
fig = plt.figure(figsize=(8, 13))
fig.suptitle('ARMA: Autocorrelation (left) and Partial Autocorrelation (right)')
subplotcount = 1
nrows = 4
for arcoef in arcoefs[:-1]:
    for macoef in macoefs[:-1]:
        ar = np.r_[1.0, -arcoef]
        ma = np.r_[1.0, macoef]
        armaprocess = tsp.ArmaProcess(ar, ma)
        acf = armaprocess.acf(20)[:20]
        pacf = armaprocess.pacf(20)[:20]
        ax = fig.add_subplot(nrows, 2, subplotcount)
        plotacf(acf, ax=ax)
        ax.text(0.7, 0.6, 'ar =%s \nma=%s' % (ar, ma), transform=ax.transAxes, horizontalalignment='left', size='xx-small')
        ax.set_xlim(-1, 20)
        subplotcount += 1
        ax = fig.add_subplot(nrows, 2, subplotcount)
        plotacf(pacf, ax=ax)
        ax.text(0.7, 0.6, 'ar =%s \nma=%s' % (ar, ma), transform=ax.transAxes, horizontalalignment='left', size='xx-small')
        ax.set_xlim(-1, 20)
        subplotcount += 1
axs = fig.axes
for ax in axs[:-2]:
    for label in ax.get_xticklabels():
        label.set_visible(False)
for ax in axs:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
plt.show()