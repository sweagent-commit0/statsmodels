"""
You can fit your LikelihoodModel using l1 regularization by changing
    the method argument and adding an argument alpha.  See code for
    details.

The Story
---------
The maximum likelihood (ML) solution works well when the number of data
points is large and the noise is small.  When the ML solution starts
"breaking", the regularized solution should do better.

The l1 Solvers
--------------
The standard l1 solver is fmin_slsqp and is included with scipy.  It
    sometimes has trouble verifying convergence when the data size is
    large.
The l1_cvxopt_cp solver is part of CVXOPT and this package needs to be
    installed separately.  It works well even for larger data sizes.
"""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)
N = len(spector_data.endog)
K = spector_data.exog.shape[1]
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()
alpha = 0.05 * N * np.ones(K)
logit_l1_res = logit_mod.fit_regularized(method='l1', alpha=alpha, acc=1e-06)
logit_l1_cvxopt_res = logit_mod.fit_regularized(method='l1_cvxopt_cp', alpha=alpha)
print('============ Results for Logit =================')
print('ML results')
print(logit_res.summary())
print('l1 results')
print(logit_l1_res.summary())
print(logit_l1_cvxopt_res.summary())
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()
alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
alpha[-1, :] = 0
mlogit_l1_res = mlogit_mod.fit_regularized(method='l1', alpha=alpha)
print(mlogit_l1_res.params)
print('============ Results for MNLogit =================')
print('ML results')
print(mlogit_res.summary())
print('l1 results')
print(mlogit_l1_res.summary())
spector_data = sm.datasets.spector.load()
X = spector_data.exog
Y = spector_data.endog
N = 50
K = X.shape[1]
logit_mod = sm.Logit(Y, X)
coeff = np.zeros((N, K))
alphas = 1 / np.logspace(-0.5, 2, N)
for n, alpha in enumerate(alphas):
    logit_res = logit_mod.fit_regularized(method='l1', alpha=alpha, trim_mode='off', QC_tol=0.1, disp=False, QC_verbose=True, acc=1e-15)
    coeff[n, :] = logit_res.params
plt.figure(1)
plt.clf()
plt.grid()
plt.title('Regularization Path')
plt.xlabel('alpha')
plt.ylabel('Parameter value')
for i in range(K):
    plt.plot(alphas, coeff[:, i], label='X' + str(i), lw=3)
plt.legend(loc='best')
plt.show()