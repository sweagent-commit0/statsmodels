"""first examples for gam and PolynomialSmoother used for debugging

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.


Note: uncomment plt.show() to display graphs
"""
import time
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt
import scipy.stats
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM
from statsmodels.genmod import families
example = 2
standardize = lambda x: (x - x.mean()) / x.std()
demean = lambda x: x - x.mean()
nobs = 500
lb, ub = (-1.0, 1.0)
x1 = R.uniform(lb, ub, nobs)
x1 = np.linspace(lb, ub, nobs)
x1.sort()
x2 = R.uniform(lb, ub, nobs)
x2.sort()
x2 = x2 + np.exp(x2 / 2.0)
y = 0.5 * R.uniform(lb, ub, nobs)
f1 = lambda x1: 2 * x1 - 0.5 * x1 ** 2 - 0.75 * x1 ** 3
f2 = lambda x2: x2 - 1 * x2 ** 2
z = standardize(f1(x1)) + standardize(f2(x2))
z = standardize(z) + 1
z = f1(x1) + f2(x2)
z -= np.median(z)
print('z.std()', z.std())
print(z.mean(), z.min(), z.max())
y = z
d = np.array([x1, x2]).T
if example == 1:
    print('normal')
    m = AdditiveModel(d)
    m.fit(y)
    x = np.linspace(-2, 2, 50)
    print(m)
if example == 2:
    print('binomial')
    mod_name = 'Binomial'
    f = families.Binomial()
    b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(z)])
    b.shape = y.shape
    m = GAM(b, d, family=f)
    toc = time.time()
    m.fit(b)
    tic = time.time()
    print(tic - toc)
    yp = f.link.inverse(y)
    p = b
if example == 3:
    print('Poisson')
    f = families.Poisson()
    yp = f.link.inverse(z)
    p = np.asarray([scipy.stats.poisson.rvs(val) for val in f.link.inverse(z)], float)
    p.shape = y.shape
    m = GAM(p, d, family=f)
    toc = time.time()
    m.fit(p)
    tic = time.time()
    print(tic - toc)
if example > 1:
    y_pred = m.results.mu
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(p, '.')
    plt.plot(yp, 'b-', label='true')
    plt.plot(y_pred, 'r-', label='GAM')
    plt.legend(loc='upper left')
    plt.title('gam.GAM ' + mod_name)
    counter = 2
    for ii, xx in zip(['z', 'x1', 'x2'], [z, x1, x2]):
        sortidx = np.argsort(xx)
        plt.subplot(2, 2, counter)
        plt.plot(xx[sortidx], p[sortidx], '.')
        plt.plot(xx[sortidx], yp[sortidx], 'b.', label='true')
        plt.plot(xx[sortidx], y_pred[sortidx], 'r.', label='GAM')
        plt.legend(loc='upper left')
        plt.title('gam.GAM ' + mod_name + ' ' + ii)
        counter += 1
    plt.figure()
    plt.plot(z, 'b-', label='true')
    plt.plot(np.log(m.results.mu), 'r-', label='GAM')
    plt.title('GAM Poisson, raw')
plt.figure()
plt.plot(x1, standardize(m.smoothers[0](x1)), 'r')
plt.plot(x1, standardize(f1(x1)), linewidth=2)
plt.figure()
plt.plot(x2, standardize(m.smoothers[1](x2)), 'r')
plt.plot(x2, standardize(f2(x2)), linewidth=2)