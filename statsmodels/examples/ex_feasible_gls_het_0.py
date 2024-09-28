"""Examples for linear model with heteroscedasticity estimated by feasible GLS

These are examples to check the results during development.

The assumptions:

We have a linear model y = X*beta where the variance of an observation depends
on some explanatory variable Z (`exog_var`).
linear_model.WLS estimated the model for a given weight matrix
here we want to estimate also the weight matrix by two step or iterative WLS

Created on Wed Dec 21 12:28:17 2011

Author: Josef Perktold

"""
import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.feasible_gls import GLSHet, GLSHet2
from statsmodels.tools.tools import add_constant
examples = ['ex1']
if 'ex1' in examples:
    nsample = 300
    sig = 0.5
    np.random.seed(9876789)
    X = np.random.randn(nsample, 3)
    X = np.column_stack((np.ones((nsample, 1)), X))
    beta = [1, 0.5, -0.5, 1.0]
    y_true2 = np.dot(X, beta)
    x1 = np.linspace(0, 1, nsample)
    gamma = np.array([1, 3.0])
    z_true = add_constant(x1)
    winv = np.dot(z_true, gamma)
    het_params = sig ** 2 * np.array([1, 3.0])
    sig2_het = sig ** 2 * winv
    weights_dgp = 1 / winv
    weights_dgp /= weights_dgp.max()
    z0 = np.zeros(nsample)
    z0[nsample * 5 // 10:] = 1
    z0 = add_constant(z0)
    z1 = add_constant(x1)
    noise = np.sqrt(sig2_het) * np.random.normal(size=nsample)
    y2 = y_true2 + noise
    X2 = X[:, [0, 2]]
    X2 = X
    res_ols = OLS(y2, X2).fit()
    print('OLS beta estimates')
    print(res_ols.params)
    print('OLS stddev of beta')
    print(res_ols.bse)
    print('\nWLS')
    mod0 = GLSHet2(y2, X2, exog_var=winv)
    res0 = mod0.fit()
    print('new version')
    mod1 = GLSHet(y2, X2, exog_var=winv)
    res1 = mod1.iterative_fit(2)
    print('WLS beta estimates')
    print(res1.params)
    print(res0.params)
    print('WLS stddev of beta')
    print(res1.bse)
    print(res1.model.weights / res1.model.weights.max())
    assert_almost_equal(res1.model.weights / res1.model.weights.max(), weights_dgp, 14)
    print('residual regression params')
    print(res1.results_residual_regression.params)
    print('scale of model ?')
    print(res1.scale)
    print('unweighted residual variance, note unweighted mean is not zero')
    print(res1.resid.var())
    doplots = True
    if doplots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x1, y2, 'o')
        plt.plot(x1, y_true2, 'b-', label='true')
        plt.plot(x1, res1.fittedvalues, 'r-', label='fwls')
        plt.plot(x1, res_ols.fittedvalues, '--', label='ols')
        plt.legend()
    mod2 = GLSHet(y2, X2, exog_var=z0)
    res2 = mod2.iterative_fit(3)
    print(res2.params)
    import statsmodels.api as sm
    z = sm.add_constant(x1 / x1.max())
    mod3 = GLSHet(y2, X2, exog_var=z1)
    res3 = mod3.iterative_fit(20)
    error_var_3 = res3.mse_resid / res3.model.weights
    print(res3.params)
    print("np.array(res3.model.history['ols_params'])")
    print(np.array(res3.model.history['ols_params']))
    print("np.array(res3.model.history['self_params'])")
    print(np.array(res3.model.history['self_params']))
    print(np.unique(res2.model.weights))
    print(np.unique(res3.model.weights))
    print(res3.summary())
    print('\n\nResults of estimation of weights')
    print('--------------------------------')
    print(res3.results_residual_regression.summary())
    if doplots:
        plt.figure()
        plt.plot(x1, y2, 'o')
        plt.plot(x1, y_true2, 'b-', label='true')
        plt.plot(x1, res1.fittedvalues, '-', label='fwls1')
        plt.plot(x1, res2.fittedvalues, '-', label='fwls2')
        plt.plot(x1, res3.fittedvalues, '-', label='fwls3')
        plt.plot(x1, res_ols.fittedvalues, '--', label='ols')
        plt.legend()
        plt.figure()
        plt.ylim(0, 5)
        res_e2 = OLS(noise ** 2, z).fit()
        plt.plot(noise ** 2, 'bo', alpha=0.5, label='dgp error**2')
        plt.plot(res_e2.fittedvalues, lw=2, label='ols for noise**2')
        plt.plot(error_var_3, lw=2, label='GLSHet error var')
        plt.plot(res3.resid ** 2, 'ro', alpha=0.5, label='resid squared')
        plt.plot(sig ** 2 * winv, lw=2, label='DGP error var')
        plt.legend()
        plt.show()
    'Note these are close but maybe biased because of skewed distribution\n    >>> res3.mse_resid/res3.model.weights[-10:]\n    array([ 1.03115871,  1.03268209,  1.03420547,  1.03572885,  1.03725223,\n            1.03877561,  1.04029899,  1.04182237,  1.04334575,  1.04486913])\n    >>> res_e2.fittedvalues[-10:]\n    array([ 1.0401953 ,  1.04171386,  1.04323242,  1.04475098,  1.04626954,\n            1.0477881 ,  1.04930666,  1.05082521,  1.05234377,  1.05386233])\n    >>> sig**2 * w[-10:]\n    array([ 0.98647295,  0.98797595,  0.98947896,  0.99098196,  0.99248497,\n            0.99398798,  0.99549098,  0.99699399,  0.99849699,  1.        ])\n        '