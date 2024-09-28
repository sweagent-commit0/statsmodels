"""Examples for linear model with heteroscedasticity estimated by feasible GLS

These are examples to check the results during development.

The assumptions:

We have a linear model y = X*beta where the variance of an observation depends
on some explanatory variable Z (`exog_var`).
linear_model.WLS estimated the model for a given weight matrix
here we want to estimate also the weight matrix by two step or iterative WLS

Created on Wed Dec 21 12:28:17 2011

Author: Josef Perktold

There might be something fishy with the example, but I do not see it.
Or maybe it's supposed to be this way because in the first case I do not
include a constant and in the second case I include some of the same
regressors as in the main equation.

"""
import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.feasible_gls import GLSHet, GLSHet2
examples = ['ex1']
if 'ex1' in examples:
    nsample = 1000
    sig = 0.5
    x1 = np.linspace(0, 20, nsample)
    X = np.c_[x1, (x1 - 5) ** 2, np.ones(nsample)]
    np.random.seed(0)
    beta = [0.5, -0.015, 1.0]
    y_true2 = np.dot(X, beta)
    w = np.ones(nsample)
    w[nsample * 6 // 10:] = 4
    y2 = y_true2 + sig * np.sqrt(w) * np.random.normal(size=nsample)
    X2 = X[:, [0, 2]]
    X2 = X
    res_ols = OLS(y2, X2).fit()
    print('OLS beta estimates')
    print(res_ols.params)
    print('OLS stddev of beta')
    print(res_ols.bse)
    print('\nWLS')
    mod0 = GLSHet2(y2, X2, exog_var=w)
    res0 = mod0.fit()
    print('new version')
    mod1 = GLSHet(y2, X2, exog_var=w)
    res1 = mod1.iterative_fit(2)
    print('WLS beta estimates')
    print(res1.params)
    print(res0.params)
    print('WLS stddev of beta')
    print(res1.bse)
    print(res1.model.weights / res1.model.weights.max())
    assert_almost_equal(res1.model.weights / res1.model.weights.max(), 1.0 / w, 14)
    print('residual regression params')
    print(res1.results_residual_regression.params)
    print('scale of model ?')
    print(res1.scale)
    print('unweighted residual variance, note unweighted mean is not zero')
    print(res1.resid.var())
    doplots = False
    if doplots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x1, y2, 'o')
        plt.plot(x1, y_true2, 'b-', label='true')
        plt.plot(x1, res1.fittedvalues, 'r-', label='fwls')
        plt.plot(x1, res_ols.fittedvalues, '--', label='ols')
        plt.legend()
    z = (w[:, None] == np.unique(w)).astype(float)
    mod2 = GLSHet(y2, X2, exog_var=z)
    res2 = mod2.iterative_fit(2)
    print(res2.params)
    import statsmodels.api as sm
    z = sm.add_constant(w)
    mod3 = GLSHet(y2, X2, exog_var=z)
    res3 = mod3.iterative_fit(8)
    print(res3.params)
    print("np.array(res3.model.history['ols_params'])")
    print(np.array(res3.model.history['ols_params']))
    print("np.array(res3.model.history['self_params'])")
    print(np.array(res3.model.history['self_params']))
    print(np.unique(res2.model.weights))
    print(np.unique(res3.model.weights))
    if doplots:
        plt.figure()
        plt.plot(x1, y2, 'o')
        plt.plot(x1, y_true2, 'b-', label='true')
        plt.plot(x1, res1.fittedvalues, '-', label='fwls1')
        plt.plot(x1, res2.fittedvalues, '-', label='fwls2')
        plt.plot(x1, res3.fittedvalues, '-', label='fwls3')
        plt.plot(x1, res_ols.fittedvalues, '--', label='ols')
        plt.legend()
        plt.show()