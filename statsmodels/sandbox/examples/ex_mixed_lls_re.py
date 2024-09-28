"""Example using OneWayMixed


Created on Sat Dec 03 10:15:55 2011

Author: Josef Perktold

This example constructs a linear model with individual specific random
effects, and uses OneWayMixed to estimate it.

This is a variation on ex_mixed_lls_0.py. Here we only have a single
individual specific constant, that is just a random effect without exogenous
regressors.

"""
import numpy as np
from statsmodels.sandbox.panel.mixed import OneWayMixed, Unit
examples = ['ex1']
if 'ex1' in examples:
    np.random.seed(978326)
    nsubj = 2000
    units = []
    nobs_i = 4
    nx = 0
    nz = 1
    beta = np.ones(nx)
    gamma = 0.5 * np.ones(nz)
    gamma[0] = 0
    gamma_re_true = []
    for i in range(nsubj):
        gamma_re = gamma + 0.2 * np.random.standard_normal(nz)
        gamma_re_true.append(gamma_re)
        if i > nsubj // 4:
            nobs_i = 6
        X = np.random.standard_normal((nobs_i, nx))
        Z = np.random.standard_normal((nobs_i, nz - 1))
        Z = np.column_stack((np.ones(nobs_i), Z))
        noise = 0.1 * np.random.randn(nobs_i)
        Y = np.dot(X, beta) + np.dot(Z, gamma_re) + noise
        X = np.hstack((X, Z))
        new_unit = Unit(Y, X, Z)
        units.append(new_unit)
    m = OneWayMixed(units)
    import time
    t0 = time.time()
    m.initialize()
    res = m.fit(maxiter=100, rtol=1e-05, params_rtol=1e-06, params_atol=1e-06)
    t1 = time.time()
    print('time for initialize and fit', t1 - t0)
    print('number of iterations', m.iterations)
    print('\nestimates for fixed effects')
    print(m.a)
    print(m.params)
    bfixed_cov = m.cov_fixed()
    print('beta fixed standard errors')
    print(np.sqrt(np.diag(bfixed_cov)))
    print(m.bse)
    b_re = m.params_random_units
    print('RE mean:', b_re.mean(0))
    print('RE columns std', b_re.std(0))
    print('np.cov(b_re, rowvar=0), sample statistic')
    print(np.cov(b_re, rowvar=0))
    print('std of above')
    print(np.sqrt(np.diag(np.atleast_1d(np.cov(b_re, rowvar=0)))))
    print('m.cov_random()')
    print(m.cov_random())
    print('std of above')
    print(res.std_random())
    print(np.sqrt(np.diag(m.cov_random())))
    print('\n(non)convergence of llf')
    print(m.history['llf'][-4:])
    print('convergence of parameters')
    print(np.diff(np.vstack(m.history['params'][-4:]), axis=0))
    print('convergence of D')
    print(np.diff(np.array(m.history['D'][-4:]), axis=0))
    zb = np.array([(unit.Z * unit.b[None, :]).sum(0) for unit in m.units])
    'if Z is not included in X:\n    >>> np.dot(b_re.T, b_re)/100\n    array([[ 0.03270611, -0.00916051],\n           [-0.00916051,  0.26432783]])\n    >>> m.cov_random()\n    array([[ 0.0348722 , -0.00909159],\n           [-0.00909159,  0.26846254]])\n    >>> #note cov_random does not subtract mean!\n    '
    print('\nchecking the random effects distribution and prediction')
    gamma_re_true = np.array(gamma_re_true)
    print('mean of random effect true', gamma_re_true.mean(0))
    print('mean from fixed effects   ', m.params[-2:])
    print('mean of estimated RE      ', b_re.mean(0))
    print()
    absmean_true = np.abs(gamma_re_true).mean(0)
    mape = ((m.params[-2:] + b_re) / gamma_re_true - 1).mean(0) * 100
    mean_abs_perc = np.abs(m.params[-2:] + b_re - gamma_re_true).mean(0) / absmean_true * 100
    median_abs_perc = np.median(np.abs(m.params[-2:] + b_re - gamma_re_true), 0) / absmean_true * 100
    rmse_perc = (m.params[-2:] + b_re - gamma_re_true).std(0) / absmean_true * 100
    print('mape           ', mape)
    print('mean_abs_perc  ', mean_abs_perc)
    print('median_abs_perc', median_abs_perc)
    print('rmse_perc (std)', rmse_perc)
    print('llf', res.llf)
    print('tvalues', res.tvalues)
    print('pvalues', res.pvalues)
    print(res.t_test([1]))
    print('test mean of both random effects variables is zero')
    print(res.f_test([[1]]))
    plots = res.plot_random_univariate(bins=50)
    import matplotlib.pyplot as plt
    plt.show()