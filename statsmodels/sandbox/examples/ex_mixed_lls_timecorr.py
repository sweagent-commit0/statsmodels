"""Example using OneWayMixed with within group intertemporal correlation


Created on Sat Dec 03 10:15:55 2011

Author: Josef Perktold

This example constructs a linear model with individual specific random
effects, and uses OneWayMixed to estimate it.

This is a variation on ex_mixed_lls_0.py.
Here we use time dummies as random effects (all except 1st time period).

I think, this should allow for (almost) arbitrary intertemporal correlation.
The assumption is that each unit can have different constants, however the
intertemporal covariance matrix is the same for all units. One caveat, to
avoid singular matrices, we have to treat one time period differently.

Estimation requires that the number of units is larger than the number of
time periods. Also, it requires that we have the same number of periods for
each unit.

I needed to remove the first observation from the time dummies to avoid a
singular matrix. So, interpretation of time effects should be relative to
first observation. (I did not check the math.)
TODO:
Note, I do not already have constant in X. Constant for first
time observation is missing.
Do I need all dummies in exog_fe, Z, but not in exog_re, Z?  Tried this and
it works.
In the error decomposition we also have the noise variable, I guess this works
like constant, so we get full rank (square) with only T-1 time dummies.
But we do not get correlation with the noise, or do we? conditional?
-> sample correlation of estimated random effects looks a bit high,
   upward bias? or still some problems with initial condition?
   correlation from estimated cov_random looks good.

Since we include the time dummies also in the fixed effect, we can have
arbitrary trends, different constants in each period.


Intertemporal correlation in data generating process, DGP, to see if
the results correctly estimate it.
used AR(1) as example, but only starting at second period. (?)
Note: we do not impose AR structure in the estimation

"""
import numpy as np
from statsmodels.sandbox.panel.mixed import OneWayMixed, Unit
examples = ['ex1']
if 'ex1' in examples:
    nsubj = 200
    units = []
    nobs_i = 8
    nx = 1
    nz = nobs_i - 1
    beta = np.ones(nx)
    gamma = 0.5 * np.ones(nz)
    gamma_re_true = []
    for i in range(nsubj):
        use_correlated = True
        if not use_correlated:
            gamma_re = gamma + 0.2 * np.random.standard_normal(nz)
        else:
            from scipy import linalg as splinalg
            rho = 0.6
            corr_re = splinalg.toeplitz(rho ** np.arange(nz))
            rvs = np.random.multivariate_normal(np.zeros(nz), corr_re)
            gamma_re = gamma + 0.2 * rvs
        gamma_re_true.append(gamma_re)
        X = np.random.standard_normal((nobs_i, nx))
        time_dummies = (np.arange(nobs_i)[:, None] == np.arange(nobs_i)[None, :]).astype(float)
        Z = time_dummies[:, 1:]
        noise = 0.1 * np.random.randn(nobs_i)
        Y = np.dot(X, beta) + np.dot(Z, gamma_re) + noise
        X = np.hstack((X, time_dummies))
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
    print('sample correlation of estimated random effects')
    print(np.corrcoef(b_re, rowvar=0))
    print('std of above')
    print(np.sqrt(np.diag(np.atleast_1d(np.cov(b_re, rowvar=0)))))
    print('m.cov_random()')
    print(m.cov_random())
    print('correlation from above')
    print(res.cov_random() / res.std_random()[:, None] / res.std_random())
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
    mape = ((m.params[-nz:] + b_re) / gamma_re_true - 1).mean(0) * 100
    mean_abs_perc = np.abs(m.params[-nz:] + b_re - gamma_re_true).mean(0) / absmean_true * 100
    median_abs_perc = np.median(np.abs(m.params[-nz:] + b_re - gamma_re_true), 0) / absmean_true * 100
    rmse_perc = (m.params[-nz:] + b_re - gamma_re_true).std(0) / absmean_true * 100
    print('mape           ', mape)
    print('mean_abs_perc  ', mean_abs_perc)
    print('median_abs_perc', median_abs_perc)
    print('rmse_perc (std)', rmse_perc)
    print('llf', res.llf)
    print('tvalues', res.tvalues)
    print('pvalues', res.pvalues)
    rmat = np.zeros(len(res.params))
    rmat[-nz:] = 1
    print('t_test mean of random effects variables are zero')
    print(res.t_test(rmat))
    print('f_test mean of both random effects variables is zero (joint hypothesis)')
    print(res.f_test(rmat))
    plots = res.plot_random_univariate()
    fig = res.plot_scatter_all_pairs()
    import matplotlib.pyplot as plt
    plt.show()