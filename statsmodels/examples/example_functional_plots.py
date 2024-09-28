"""Functional boxplots and rainbow plots

see docstrings for an explanation


Author: Ralf Gommers

"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
data = sm.datasets.elnino.load()
fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.fboxplot(data.raw_data[:, 1:], wfactor=2.58, labels=data.raw_data[:, 0].astype(int), ax=ax)
ax.set_xlabel('Month of the year')
ax.set_ylabel('Sea surface temperature (C)')
ax.set_xticks(np.arange(13, step=3) - 1)
ax.set_xticklabels(['', 'Mar', 'Jun', 'Sep', 'Dec'])
ax.set_xlim([-0.2, 11.2])
fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.rainbowplot(data.raw_data[:, 1:], ax=ax)
ax.set_xlabel('Month of the year')
ax.set_ylabel('Sea surface temperature (C)')
ax.set_xticks(np.arange(13, step=3) - 1)
ax.set_xticklabels(['', 'Mar', 'Jun', 'Sep', 'Dec'])
ax.set_xlim([-0.2, 11.2])
plt.show()