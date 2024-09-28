"""Examples for Regression Plots

Author: Josef Perktold

"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.graphics.regressionplots as smrp
from statsmodels.graphics.tests.test_regressionplots import TestPlot
np.random.seed(9876789)
nsample = 100
sig = 0.5
x1 = np.linspace(0, 20, nsample)
x2 = 5 + 3 * np.random.randn(nsample)
X = np.c_[x1, x2, np.sin(0.5 * x1), (x2 - 5) ** 2, np.ones(nsample)]
beta = [0.5, 0.5, 1, -0.04, 5.0]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)
exog0 = sm.add_constant(np.c_[x1, x2], prepend=False)
res = sm.OLS(y, exog0).fit()
plot_old = 0
if plot_old:
    prstd, iv_l, iv_u = wls_prediction_std(res)
    plt.plot(x1, res.fittedvalues, 'r-o')
    plt.plot(x1, iv_u, 'r--')
    plt.plot(x1, iv_l, 'r--')
    plt.title('blue: true,   red: OLS')
    plt.figure()
    plt.plot(res.resid, 'o')
    plt.title('Residuals')
    fig2 = plt.figure()
    ax = fig2.add_subplot(2, 1, 1)
    plt.plot(x1, res.resid, 'o')
    ax.set_title('residuals versus exog')
    ax = fig2.add_subplot(2, 1, 2)
    plt.plot(x2, res.resid, 'o')
    fig3 = plt.figure()
    ax = fig3.add_subplot(2, 1, 1)
    plt.plot(x1, res.fittedvalues, 'o')
    ax.set_title('Fitted values versus exog')
    ax = fig3.add_subplot(2, 1, 2)
    plt.plot(x2, res.fittedvalues, 'o')
    fig4 = plt.figure()
    ax = fig4.add_subplot(2, 1, 1)
    plt.plot(x1, res.fittedvalues + res.resid, 'o')
    ax.set_title('Fitted values plus residuals versus exog')
    ax = fig4.add_subplot(2, 1, 2)
    plt.plot(x2, res.fittedvalues + res.resid, 'o')
    fig5 = plt.figure()
    ax = fig5.add_subplot(2, 1, 1)
    res1a = sm.OLS(y, exog0[:, [0, 2]]).fit()
    res1b = sm.OLS(x1, exog0[:, [0, 2]]).fit()
    plt.plot(res1b.resid, res1a.resid, 'o')
    res1c = sm.OLS(res1a.resid, res1b.resid).fit()
    plt.plot(res1b.resid, res1c.fittedvalues, '-')
    ax.set_title('Partial Regression plot')
    ax = fig5.add_subplot(2, 1, 2)
    res2a = sm.OLS(y, exog0[:, [0, 1]]).fit()
    res2b = sm.OLS(x2, exog0[:, [0, 1]]).fit()
    plt.plot(res2b.resid, res2a.resid, 'o')
    res2c = sm.OLS(res2a.resid, res2b.resid).fit()
    plt.plot(res2b.resid, res2c.fittedvalues, '-')
    fig6 = plt.figure()
    ax = fig6.add_subplot(2, 1, 1)
    x1beta = x1 * res.params[1]
    x2beta = x2 * res.params[2]
    plt.plot(x1, x1beta + res.resid, 'o')
    plt.plot(x1, x1beta, '-')
    ax.set_title('X_i beta_i plus residuals versus exog (CCPR)')
    ax = fig6.add_subplot(2, 1, 2)
    plt.plot(x2, x2beta + res.resid, 'o')
    plt.plot(x2, x2beta, '-')
doplots = 1
if doplots:
    fig1 = smrp.plot_fit(res, 0, y_true=None)
    smrp.plot_fit(res, 1, y_true=None)
    smrp.plot_partregress_grid(res, exog_idx=[0, 1])
    smrp.plot_regress_exog(res, exog_idx=0)
    smrp.plot_ccpr(res, exog_idx=0)
    smrp.plot_ccpr_grid(res, exog_idx=[0, 1])
tp = TestPlot()
tp.test_plot_fit()
fig1 = smrp.plot_partregress_grid(res, exog_idx=[0, 1])
ax = fig1.axes[0]
y0 = ax.get_lines()[0]._y
x0 = ax.get_lines()[0]._x
lres = sm.nonparametric.lowess(y0, x0, frac=0.2)
ax.plot(lres[:, 0], lres[:, 1], 'r', lw=1.5)
ax = fig1.axes[1]
y0 = ax.get_lines()[0]._y
x0 = ax.get_lines()[0]._x
lres = sm.nonparametric.lowess(y0, x0, frac=0.2)
ax.plot(lres[:, 0], lres[:, 1], 'r', lw=1.5)