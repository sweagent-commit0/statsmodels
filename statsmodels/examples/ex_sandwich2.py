"""Cluster robust standard errors for OLS

Created on Fri Dec 16 12:52:13 2011
Author: Josef Perktold
"""
from urllib.request import urlretrieve
import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw
import statsmodels.iolib.foreign as dta
try:
    srs = dta.genfromdta('srs.dta')
    print('using local file')
except IOError:
    urlretrieve('http://www.ats.ucla.edu/stat/stata/seminars/svy_stata_intro/srs.dta', 'srs.dta')
    print('downloading file')
    srs = dta.genfromdta('srs.dta')
y = srs['api00']
x = np.column_stack([srs[ii] for ii in ['growth', 'emer', 'yr_rnd']])
group = srs['dnum']
xx = sm.add_constant(x, prepend=False)
mask = (xx != -999.0).all(1)
mask.shape
y = y[mask]
xx = xx[mask]
group = group[mask]
res_srs = sm.OLS(y, xx).fit()
print('params    ', res_srs.params)
print('bse_OLS   ', res_srs.bse)
cov_cr = sw.cov_cluster(res_srs, group.astype(int))
bse_cr = sw.se_cov(cov_cr)
print('bse_rob   ', bse_cr)
res_stata = np.rec.array([('growth', '|', -0.1027121, 0.2291703, -0.45, 0.655, -0.5548352, 0.3494111), ('emer', '|', -5.444932, 0.7293969, -7.46, 0.0, -6.883938, -4.005927), ('yr_rnd', '|', -51.07569, 22.83615, -2.24, 0.027, -96.12844, -6.022935), ('_cons', '|', 740.3981, 13.46076, 55.0, 0.0, 713.8418, 766.9544)], dtype=[('exogname', '|S6'), ('del', '|S1'), ('params', 'float'), ('bse', 'float'), ('tvalues', 'float'), ('pvalues', 'float'), ('cilow', 'float'), ('ciupp', 'float')])
print('diff Stata', bse_cr - res_stata.bse)
assert_almost_equal(bse_cr, res_stata.bse, decimal=6)
print('reldiff to OLS', bse_cr / res_srs.bse - 1)