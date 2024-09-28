"""Examples OLS

Note: uncomment plt.show() to display graphs

Summary:
========

Relevant part of construction of design matrix
xg includes group numbers/labels,
x1 is continuous explanatory variable

>>> dummy = (xg[:,None] == np.unique(xg)).astype(float)
>>> X = np.c_[x1, dummy[:,1:], np.ones(nsample)]

Estimate the model

>>> res2 = sm.OLS(y, X).fit()
>>> print res2.params
[ 1.00901524  3.08466166 -2.84716135  9.94655423]
>>> print res2.bse
[ 0.07499873  0.71217506  1.16037215  0.38826843]
>>> prstd, iv_l, iv_u = wls_prediction_std(res2)

"Test hypothesis that all groups have same intercept"

>>> R = [[0, 1, 0, 0],
...      [0, 0, 1, 0]]

>>> print res2.f_test(R)
<F test: F=array([[ 91.69986847]]), p=[[  8.90826383e-17]], df_denom=46, df_num=2>

strongly rejected because differences in intercept are very large

"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
np.random.seed(98765789)
nsample = 50
x1 = np.linspace(0, 20, nsample)
sig = 1.0
xg = np.zeros(nsample, int)
xg[20:40] = 1
xg[40:] = 2
dummy = (xg[:, None] == np.unique(xg)).astype(float)
X = np.c_[x1, dummy[:, 1:], np.ones(nsample)]
beta = [1.0, 3, -3, 10]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)
res2 = sm.OLS(y, X).fit()
print(res2.params)
print(res2.bse)
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.figure()
plt.plot(x1, y, 'o', x1, y_true, 'b-')
plt.plot(x1, res2.fittedvalues, 'r--.')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
plt.title('3 groups: different intercepts, common slope; blue: true, red: OLS')
plt.show()
R = [[0, 1, 0, 0], [0, 0, 1, 0]]
print('Test hypothesis that all groups have same intercept')
print(res2.f_test(R))