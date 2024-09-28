"""calculating anova and verifying with NIST test data

compares my implementations, stats.f_oneway and anova using statsmodels.OLS
"""
from statsmodels.compat.python import lmap
import os
import numpy as np
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from .try_ols_anova import data2dummy
filenameli = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat', 'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat', 'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']
if __name__ == '__main__':
    print('\n using new ANOVA anova_oneway')
    print('f, prob, R2, resstd')
    for fn in filenameli:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        res = anova_oneway(y, x)
        rtol = {'SmLs08.dat': 0.027, 'SmLs07.dat': 0.0017, 'SmLs09.dat': 0.0001}.get(fn, 1e-07)
        np.testing.assert_allclose(np.array(res), cert, rtol=rtol)
    print('\n using stats ANOVA f_oneway')
    for fn in filenameli:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        xlist = [x[y == ii] for ii in caty]
        res = stats.f_oneway(*xlist)
        print(np.array(res) - cert[:2])
    print('\n using statsmodels.OLS')
    print('f, prob, R2, resstd')
    for fn in filenameli[:]:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        res = anova_ols(x, y)
        print(np.array(res) - cert)