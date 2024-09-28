"""Example: Test for equality of coefficients across groups/regressions


Created on Sat Mar 27 22:36:51 2010
Author: josef-pktd
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.onewaygls import OneWayLS
example = ['null', 'diff'][1]
example_size = [10, 100][0]
example_size = [(10, 2), (100, 2)][0]
example_groups = ['2', '2-2'][1]
np.random.seed(87654589)
nobs, nvars = example_size
x1 = np.random.normal(size=(nobs, nvars))
y1 = 10 + np.dot(x1, [15.0] * nvars) + 2 * np.random.normal(size=nobs)
x1 = sm.add_constant(x1, prepend=False)
x2 = np.random.normal(size=(nobs, nvars))
if example == 'null':
    y2 = 10 + np.dot(x2, [15.0] * nvars) + 2 * np.random.normal(size=nobs)
else:
    y2 = 19 + np.dot(x2, [17.0] * nvars) + 2 * np.random.normal(size=nobs)
x2 = sm.add_constant(x2, prepend=False)
x = np.concatenate((x1, x2), 0)
y = np.concatenate((y1, y2))
if example_groups == '2':
    groupind = (np.arange(2 * nobs) > nobs - 1).astype(int)
else:
    groupind = np.mod(np.arange(2 * nobs), 4)
    groupind.sort()
print('\nTest for equality of coefficients for all exogenous variables')
print('-------------------------------------------------------------')
res = OneWayLS(y, x, groups=groupind.astype(int))
print_results(res)
print('\n\nOne way ANOVA, constant is the only regressor')
print('---------------------------------------------')
print('this is the same as scipy.stats.f_oneway')
res = OneWayLS(y, np.ones(len(y)), groups=groupind)
print_results(res)
print('\n\nOne way ANOVA, constant is the only regressor with het is true')
print('--------------------------------------------------------------')
print('this is the similar to scipy.stats.f_oneway,')
print('but variance is not assumed to be the same across groups')
res = OneWayLS(y, np.ones(len(y)), groups=groupind.astype(str), het=True)
print_results(res)
print(res.print_summary())